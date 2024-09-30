import os
import torch
from random import randint
import random
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim,lncc
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy, rgb_to_gray
from lib.utils.mask_utils import auto_illu_mask, hist_mask
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from SCI.model import SCINetwork
from lib.utils.img_utils import get_mcie_relight
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
import time
import numpy as np
import cv2
import ipdb
import torch.nn.functional as F
import matplotlib.pyplot as plt  # 用于绘图和保存图像

from utils.graphics_utils import patch_offsets, patch_warp
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam


def training():
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    start_iter = 0
    tb_writer = prepare_output_and_logger()
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)

    loss_df = pd.DataFrame(columns=['Iteration', 'L1_loss','sci_loss','sky_loss','semantic_loss','obj_acc_loss','lidar_depth_loss',
                                    'color_correction_reg_loss','pose_correction_reg_loss','scale_flatten_loss','sparse_loss','normal_loss','geo_loss','ncc_loss'])
    
    scene = Scene(gaussians=gaussians, dataset=dataset)
    sci_enhance = SCINetwork(stage=3)
    
    sci_enhance.enhance.in_conv.apply(sci_enhance.weights_init)
    sci_enhance.enhance.blocks.apply(sci_enhance.weights_init)
    sci_enhance.enhance.out_conv.apply(sci_enhance.weights_init)
    sci_enhance.calibrate.in_conv.apply(sci_enhance.weights_init)
    sci_enhance.calibrate.blocks.apply(sci_enhance.weights_init)
    sci_enhance.calibrate.out_conv.apply(sci_enhance.weights_init)
    sci_enhance = sci_enhance.cuda()
    sci_optimizer = torch.optim.Adam(sci_enhance.parameters(), lr=optim_args.sci_lr, betas=(0.9, 0.999), weight_decay=3e-4)
    
    gaussians.training_setup()
    try:
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
        state_dict = torch.load(ckpt_path)
        start_iter = state_dict['iter']
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)
    except:
        pass

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    viewpoint_stack = None
    for iteration in range(start_iter, training_args.iterations + 1):
    
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Get the unenhanced ground truth image from viewpoint camera
        gt_image_unenhance = viewpoint_cam.original_image.cuda()

        # Convert the image to grayscale
        gray_image = rgb_to_gray(gt_image_unenhance)
        gray_image = gray_image.unsqueeze(0)

        # Apply SCI enhancement and generate SCI loss
        sci_loss, illu_list, i_k = sci_enhance._loss(gray_image, 0)

        # Get illumination map
        illu = illu_list[0]
        illu = torch.cat((illu, illu, illu), dim=1)  # Duplicate channels to match RGB format
        illu = illu.squeeze(0)

        # Create illumination mask
        illu_mask, k = auto_illu_mask(illu)
        illu_mask = torch.floor(illu_mask).to(torch.bool)

        # Enhance original image using the illumination map
        r = gt_image_unenhance / illu 
        r = torch.clamp(r, 0, 1)

        # Expand illumination mask to match the shape of the original image
        illu_mask = illu_mask.expand_as(gt_image_unenhance)

        # Apply histogram equalization and denoising
        gt_image_unhist = r.squeeze(0)
        gt_image_undenoise = get_mcie_relight(gt_image_unhist)
        gt_image_undenoise = gt_image_undenoise.squeeze(0)

        # Convert tensor to NumPy for OpenCV processing
        gt_image_undenoise_np = gt_image_undenoise.permute(1, 2, 0).cpu().numpy()
        gt_image_undenoise_np = (gt_image_undenoise_np * 255).astype(np.uint8)

        # Denoise the image using OpenCV's fastNlMeansDenoisingColored
        gt_image_unmask = cv2.fastNlMeansDenoisingColored(gt_image_undenoise_np, None, 10, 10, 7, 21)

        # Convert back to tensor and scale to [0, 1]
        gt_image_unmask = torch.tensor(gt_image_unmask).float().cuda() / 255.0
        gt_image_unmask = gt_image_unmask.permute(2, 0, 1)

        # Apply darkening factor to overexposed areas
        darkening_factor = 0.5
        gt_image = gt_image_unmask * (illu_mask) + (gt_image_unmask * darkening_factor) * (~illu_mask)

        # Ensure the pixel values are within [0, 1]
        gt_image = torch.clamp(gt_image, 0, 1)
    
        # Visualization and processing during training
        if iteration % 1000 == 0:
            save_dir = os.path.join(cfg.trained_model_dir, 'illu_images')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Convert tensors to NumPy arrays for visualization
            illu_mask_np = illu_mask.detach().cpu().numpy().astype(np.float32)
            illu_mask_np = np.transpose(illu_mask_np, (1, 2, 0))

            r_np = r.detach().cpu().numpy().astype(np.float32)
            r_np = np.transpose(r_np, (1, 2, 0))

            gt_image_unenhance_np = gt_image_unenhance.detach().cpu().numpy().astype(np.float32)
            gt_image_unenhance_np = np.transpose(gt_image_unenhance_np, (1, 2, 0))

            gt_image_np = gt_image.detach().cpu().numpy().astype(np.float32)
            gt_image_np = np.transpose(gt_image_np, (1, 2, 0))

            gt_image_undenoise_np2 = gt_image_undenoise.detach().cpu().numpy().astype(np.float32)
            gt_image_undenoise_np2 = np.transpose(gt_image_undenoise_np2, (1, 2, 0))

            gt_image_unmask_np = gt_image_unmask.detach().cpu().numpy().astype(np.float32)
            gt_image_unmask_np = np.transpose(gt_image_unmask_np, (1, 2, 0))

            # Create a figure for visualization
            fig, axes = plt.subplots(1, 6, figsize=(48, 8))

            # Display the original unenhanced image
            axes[0].imshow(gt_image_unenhance_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Display the illumination mask
            axes[1].imshow(illu_mask_np, cmap='gray')
            axes[1].set_title('Illuminance Mask')
            axes[1].axis('off')

            # Display the enhanced image
            axes[2].imshow(r_np)
            axes[2].set_title('Enhanced Image (SCI)')
            axes[2].axis('off')

            # Display the denoised image
            axes[3].imshow(gt_image_undenoise_np2)
            axes[3].set_title('Denoised Image')
            axes[3].axis('off')

            # Display the unmasked image after enhancement
            axes[4].imshow(gt_image_unmask_np)
            axes[4].set_title('Unmasked Image')
            axes[4].axis('off')

            # Display the final image after all processing
            axes[5].imshow(gt_image_np)
            axes[5].set_title('Final Processed Image')
            axes[5].axis('off')

            # Save the figure
            save_path = os.path.join(save_dir, f'iteration_{iteration}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved image for iteration {iteration} at {save_path}")

        # Check if the viewpoint camera has an 'original_mask' attribute; if not, create a default mask
        if hasattr(viewpoint_cam, 'original_mask'):
            mask = viewpoint_cam.original_mask.cuda().bool()
        else:
            # Default to a mask of ones if no original mask is available, ensuring all pixels are considered
            mask = torch.ones_like(gt_image[0:1]).bool()

        # Check if the viewpoint camera has an 'original_sky_mask' attribute
        if hasattr(viewpoint_cam, 'original_sky_mask'):
            # Use the original sky mask for rendering if available
            sky_mask = viewpoint_cam.original_sky_mask.cuda()
        else:
            # Set sky_mask to None if not provided, meaning no specific sky masking will be applied
            sky_mask = None

        # Check if the viewpoint camera has an 'original_obj_bound' attribute
        if hasattr(viewpoint_cam, 'original_obj_bound'):
            # Utilize the object boundary mask to identify and separate objects in the scene
            obj_bound = viewpoint_cam.original_obj_bound.cuda().bool()
        else:
            # Default to a zero-filled mask if object boundaries are not specified
            obj_bound = torch.zeros_like(gt_image[0:1]).bool()

        # Enable debug mode for rendering if the current iteration matches the debug start iteration
        if (iteration - 1) == training_args.debug_from:
            cfg.render.debug = True  # Activates detailed rendering outputs for debugging purposes

        # Render the scene using Gaussian Renderer with the specified viewpoint camera
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, return_plane=True, return_depth_normal=True)

        # Extract rendering outputs: RGB image, accumulation map, 3D viewspace points, visibility filter, and Gaussian radii
        image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg['depth']  # Extract depth map from the rendered package for subsequent processing

        # Reset gradients for the SCI optimizer to prepare for backpropagation in this iteration
        sci_optimizer.zero_grad()



        scalar_dict = dict()
        # rgb loss
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict['l1_loss'] = Ll1.item()
        L1_loss=optim_args.lambda_l1 * Ll1
        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))

        #sci loss
        sci_loss += optim_args.lambda_sci * sci_loss
        loss += sci_loss


        # sky loss
        if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(optim_args.lambda_sky_scale) > 0:
                sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
            scalar_dict['sky_loss'] = sky_loss.item()
            sky_loss=optim_args.lambda_sky * sky_loss
            loss += sky_loss
        else: 
            sky_loss = None

        # semantic loss
        if optim_args.lambda_semantic > 0 and data_args.get('use_semantic', False) and 'semantic' in viewpoint_cam.meta:
            gt_semantic = viewpoint_cam.meta['semantic'].cuda().long() 
            if torch.all(gt_semantic == -1):
                semantic_loss = torch.zeros_like(Ll1)
            else:
                semantic = render_pkg['semantic'].unsqueeze(0) 
                semantic_loss = torch.nn.functional.cross_entropy(
                    input=semantic, 
                    target=gt_semantic,
                    ignore_index=-1, 
                    reduction='mean'
                )
            scalar_dict['semantic_loss'] = semantic_loss.item()
            semantic_loss += optim_args.lambda_semantic * semantic_loss
            loss += semantic_loss
        else : 
            semantic_loss = None
        
        # obj_acc_loss
        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
            obj_acc_loss = torch.where(obj_bound, 
                -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)), 
                -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            obj_acc_loss += optim_args.lambda_reg * obj_acc_loss
            loss += obj_acc_loss
        else : 
            obj_acc_loss = None
        
        # lidar depth loss
        if optim_args.lambda_depth_lidar > 0 and 'lidar_depth' in viewpoint_cam.meta:            
            lidar_depth = viewpoint_cam.meta['lidar_depth'].cuda() # [1, H, W]
            depth_mask = torch.logical_and((lidar_depth > 0.), mask)
            # depth_mask[obj_bound] = False
            if torch.nonzero(depth_mask).any():
                expected_depth = depth / (render_pkg['acc'] + 1e-10)  
                depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
                depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                lidar_depth_loss = depth_error.mean()
                scalar_dict['lidar_depth_loss'] = lidar_depth_loss
            else:
                lidar_depth_loss = torch.zeros_like(Ll1) 

            lidar_depth_loss += optim_args.lambda_depth_lidar * lidar_depth_loss    
            loss += lidar_depth_loss
        else : 
            lidar_depth_loss = None
                    
        # color correction loss
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            color_correction_reg_loss += optim_args.lambda_color_correction * color_correction_reg_loss
            loss += color_correction_reg_loss
        else:
            color_correction_reg_loss=None
        
        # pose correction loss
        if optim_args.lambda_pose_correction > 0 and gaussians.use_pose_correction:
            pose_correction_reg_loss = gaussians.pose_correction.regularization_loss()
            scalar_dict['pose_correction_reg_loss'] = pose_correction_reg_loss.item()
            pose_correction_reg_loss += optim_args.lambda_pose_correction * pose_correction_reg_loss
            loss += pose_correction_reg_loss
        else:
            pose_correction_reg_loss=None
                    
        # scale flatten loss
        if optim_args.lambda_scale_flatten > 0:
            scale_flatten_loss = gaussians.background.scale_flatten_loss()
            scalar_dict['scale_flatten_loss'] = scale_flatten_loss.item()
            scale_flatten_loss += optim_args.lambda_scale_flatten * scale_flatten_loss
            loss += scale_flatten_loss
        else:
            scale_flatten_loss=None

        # opacity sparse loss
        if optim_args.lambda_opacity_sparse > 0:
            opacity = gaussians.get_opacity
            opacity = opacity.clamp(1e-6, 1-1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
            sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
            scalar_dict['opacity_sparse_loss'] = sparse_loss.item()
            sparse_loss += optim_args.lambda_opacity_sparse * sparse_loss
            loss += sparse_loss
        else:
            sparse_loss=None
                
        # normal loss
        if optim_args.lambda_normal_mono > 0 and 'mono_normal' in viewpoint_cam.meta and 'normals' in render_pkg:
            if sky_mask is None:
                normal_mask = mask
            else:
                normal_mask = torch.logical_and(mask, ~sky_mask)
                normal_mask = normal_mask.squeeze(0)
                normal_mask[:50] = False
                
            normal_gt = viewpoint_cam.meta['mono_normal'].permute(1, 2, 0).cuda() 
            R_c2w = viewpoint_cam.world_view_transform[:3, :3]
            normal_gt = torch.matmul(normal_gt, R_c2w.T) 
            normal_pred = render_pkg['normals'].permute(1, 2, 0) 
            
            normal_l1_loss = torch.abs(normal_pred[normal_mask] - normal_gt[normal_mask]).mean()
            normal_cos_loss = (1. - torch.sum(normal_pred[normal_mask] * normal_gt[normal_mask], dim=-1)).mean()
            scalar_dict['normal_l1_loss'] = normal_l1_loss.item()
            scalar_dict['normal_cos_loss'] = normal_cos_loss.item()
            normal_loss = normal_l1_loss + normal_cos_loss
            normal_loss += optim_args.lambda_normal_mono * normal_loss
            loss += normal_loss
        else:
            normal_loss=None
        
        # ====================================================================
        # Multi-view consistency loss calculation
        if iteration > optim_args.multi_view_weight_from_iter:
            # Select a random nearest camera or generate a virtual camera based on probability
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id, 1)[0]]
            use_virtual_cam = False
            if optim_args.use_virtual_cam and (np.random.random() < optim_args.virtual_cam_prob or nearest_cam is None):
                # Generate a virtual camera using defined noise parameters for translation and rotation
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtual_cam = True

            if nearest_cam is not None:
                # Configuration parameters for multi-view loss calculation
                patch_size = optim_args.multi_view_patch_size
                sample_num = optim_args.multi_view_sample_num
                pixel_noise_th = optim_args.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = optim_args.multi_view_ncc_weight
                geo_weight = optim_args.multi_view_geo_weight

                # Prepare pixel coordinates and render the nearest camera view
                H, W = render_pkg['depth'].squeeze().shape
                ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['depth'].device)
                nearest_render_pkg = gaussians_renderer.render(nearest_cam, gaussians, 
                                                            return_plane=True, return_depth_normal=False)
                
                # Project depth points from the viewpoint camera to the nearest camera's space
                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3] + nearest_cam.world_view_transform[3, :3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['depth'], pts_in_nearest_cam)
                
                # Normalize and re-project points to ensure accurate depth mapping
                pts_in_nearest_cam /= pts_in_nearest_cam[:, 2:3]
                pts_in_nearest_cam *= map_z.squeeze()[..., None]

                # Apply inverse transformation to convert back to the original viewpoint camera space
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_transformed = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
                pts_in_view_cam = pts_transformed @ viewpoint_cam.world_view_transform[:3, :3] + viewpoint_cam.world_view_transform[3, :3]
                
                # Project points to the image plane using intrinsic camera parameters
                pts_projections = torch.stack(
                    [pts_in_view_cam[:, 0] * viewpoint_cam.Fx / pts_in_view_cam[:, 2] + viewpoint_cam.Cx,
                    pts_in_view_cam[:, 1] * viewpoint_cam.Fy / pts_in_view_cam[:, 2] + viewpoint_cam.Cy], -1).float()
                
                # Calculate pixel noise and construct geometry consistency mask
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                d_mask &= (pixel_noise < pixel_noise_th)
                weights = (1.0 / torch.exp(pixel_noise)).detach()  # Weight inversely proportional to pixel noise
                weights[~d_mask] = 0  # Set weights to zero where the mask is not valid

                # Compute geometric consistency loss if valid points exist
                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()  # Weighted mean of pixel noise
                    loss += geo_loss
                    ncc_loss = None  # Placeholder, NCC loss could be added later
                else:
                    geo_loss = None
                    ncc_loss = None
        else:
            geo_loss = None
            ncc_loss = None


            
        scalar_dict['loss'] = loss.item()

        loss.backward()
        
        
        iter_end.record()
                
        is_save_images = True
        if is_save_images and (iteration % 1000 == 0):
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = acc_obj.repeat(3, 1, 1)
            row1 = torch.cat([acc, image_obj, acc_obj], dim=2)
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
        
        with torch.no_grad():
            # Log
            tensor_dict = dict()

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
            if viewpoint_cam.id not in psnr_dict:
                psnr_dict[viewpoint_cam.id] = psnr(image, gt_image, mask).mean().float()
            else:
                psnr_dict[viewpoint_cam.id] = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * psnr_dict[viewpoint_cam.id]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == training_args.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                gaussians.parse_camera(viewpoint_cam)   
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                        )

                        scalar_dict.update(scalars)
                        tensor_dict.update(tensors)
                        
            # Reset opacity
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer)

            # Optimizer step
            if iteration < training_args.iterations:
                gaussians.update_optimizer()
                sci_optimizer.step()

            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)


            # Output current loss values and save them periodically
            if iteration > 0 and iteration % 10 == 0:
                print(f"Iteration {iteration}: L1_loss={L1_loss}, sci_loss={sci_loss}, sky_loss={sky_loss}, semantic_loss={semantic_loss}, "
                    f"obj_acc_loss={obj_acc_loss}, lidar_depth_loss={lidar_depth_loss}, color_correction_reg_loss={color_correction_reg_loss}, "
                    f"pose_correction_reg_loss={pose_correction_reg_loss}, scale_flatten_loss={scale_flatten_loss}, sparse_loss={sparse_loss}, "
                    f"normal_loss={normal_loss}, geo_loss={geo_loss}, ncc_loss={ncc_loss}")

            # Create directory for saving loss logs if not already present
            save_loss_dir = os.path.join(cfg.trained_model_dir, 'loss_dict')
            if not os.path.exists(save_loss_dir):
                os.makedirs(save_loss_dir)

            # Record loss values every 10 iterations
            if iteration > 0 and iteration % 10 == 0:
                # Create a new row for the current loss values
                new_row = pd.DataFrame({
                    'Iteration': [iteration],
                    'L1_loss': [L1_loss.cpu().item() if isinstance(L1_loss, torch.Tensor) else L1_loss],
                    'sci_loss': [sci_loss.cpu().item() if isinstance(sci_loss, torch.Tensor) else sci_loss],
                    'sky_loss': [sky_loss.cpu().item() if isinstance(sky_loss, torch.Tensor) else sky_loss],
                    'semantic_loss': [semantic_loss.cpu().item() if isinstance(semantic_loss, torch.Tensor) else semantic_loss],
                    'obj_acc_loss': [obj_acc_loss.cpu().item() if isinstance(obj_acc_loss, torch.Tensor) else obj_acc_loss],
                    'lidar_depth_loss': [lidar_depth_loss.cpu().item() if isinstance(lidar_depth_loss, torch.Tensor) else lidar_depth_loss],
                    'color_correction_reg_loss': [color_correction_reg_loss.cpu().item() if isinstance(color_correction_reg_loss, torch.Tensor) else color_correction_reg_loss],
                    'pose_correction_reg_loss': [pose_correction_reg_loss.cpu().item() if isinstance(pose_correction_reg_loss, torch.Tensor) else pose_correction_reg_loss],
                    'scale_flatten_loss': [scale_flatten_loss.cpu().item() if isinstance(scale_flatten_loss, torch.Tensor) else scale_flatten_loss],
                    'sparse_loss': [sparse_loss.cpu().item() if isinstance(sparse_loss, torch.Tensor) else sparse_loss],
                    'normal_loss': [normal_loss.cpu().item() if isinstance(normal_loss, torch.Tensor) else normal_loss],
                    'geo_loss': [geo_loss.cpu().item() if isinstance(geo_loss, torch.Tensor) else geo_loss],
                    'ncc_loss': [ncc_loss.cpu().item() if isinstance(ncc_loss, torch.Tensor) else ncc_loss]
                })

                # Append the new row to the loss DataFrame
                loss_df = pd.concat([loss_df, new_row], ignore_index=True)

                # Save the loss values to a text file every 1000 iterations
                if iteration % 1000 == 0:
                    save_path = os.path.join(save_loss_dir, f'loss_{iteration}.txt')
                    with open(save_path, 'w') as f:
                        for _, row in loss_df.iterrows():
                            f.write(f"Iteration {int(row['Iteration'])}: L1_loss={row['L1_loss']}, sci_loss={row['sci_loss']}, sky_loss={row['sky_loss']}, "
                                    f"semantic_loss={row['semantic_loss']}, obj_acc_loss={row['obj_acc_loss']}, lidar_depth_loss={row['lidar_depth_loss']}, "
                                    f"color_correction_reg_loss={row['color_correction_reg_loss']}, pose_correction_reg_loss={row['pose_correction_reg_loss']}, "
                                    f"scale_flatten_loss={row['scale_flatten_loss']}, sparse_loss={row['sparse_loss']}, normal_loss={row['normal_loss']}, "
                                    f"geo_loss={row['geo_loss']}, ncc_loss={row['ncc_loss']} \n")

                    # Clear the loss DataFrame for the next interval
                    loss_df = pd.DataFrame(columns=[
                        'Iteration', 
                        'L1_loss', 
                        'sci_loss', 
                        'sky_loss', 
                        'semantic_loss', 
                        'obj_acc_loss', 
                        'lidar_depth_loss',
                        'color_correction_reg_loss', 
                        'pose_correction_reg_loss', 
                        'scale_flatten_loss', 
                        'sparse_loss', 
                        'normal_loss',
                        'geo_loss',
                        'ncc_loss'
                    ])






def prepare_output_and_logger():
    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    # All done
    print("\nTraining complete.")