# import os
# import random
# import json
# import torch
# import numpy as np
# from lib.utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
# from lib.config import cfg
# from lib.datasets.base_readers import storePly, SceneInfo
# from lib.datasets.colmap_readers import readColmapSceneInfo
# from lib.datasets.blender_readers import readNerfSyntheticInfo
# from lib.datasets.waymo_full_readers import readWaymoFullInfo

# sceneLoadTypeCallbacks = {
#     "Colmap": readColmapSceneInfo,
#     "Blender" : readNerfSyntheticInfo,
#     "Waymo": readWaymoFullInfo,
# }

# class Dataset():
#     def __init__(self):
#         self.cfg = cfg.data
#         self.model_path = cfg.model_path
#         self.source_path = cfg.source_path
#         self.images = self.cfg.images

#         self.train_cameras = {}
#         self.test_cameras = {}

#         dataset_type = cfg.data.get('type', "Colmap")
#         assert dataset_type in sceneLoadTypeCallbacks.keys(), 'Could not recognize scene type!'
        
#         scene_info: SceneInfo = sceneLoadTypeCallbacks[dataset_type](self.source_path, **cfg.data)

#         if cfg.mode == 'train':
#             print(f'Saving input pointcloud to {os.path.join(self.model_path, "input.ply")}')
#             pcd = scene_info.point_cloud
#             storePly(os.path.join(self.model_path, "input.ply"), pcd.points, pcd.colors)

#             json_cams = []
#             camlist = []
#             if scene_info.test_cameras:
#                 camlist.extend(scene_info.test_cameras)
#             if scene_info.train_cameras:
#                 camlist.extend(scene_info.train_cameras)
#             for id, cam in enumerate(camlist):
#                 json_cams.append(camera_to_JSON(id, cam))

#             print(f'Saving input camera to {os.path.join(self.model_path, "cameras.json")}')
#             with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
#                 json.dump(json_cams, file)
       
#         self.scene_info = scene_info
        
#         if self.cfg.shuffle and cfg.mode == 'train':
#             random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
#             random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling
        
#         self.cameras_extent = scene_info.nerf_normalization["radius"]
#         print(f"cameras_extent {self.cameras_extent}")

#         self.multi_view_num = cfg.multi_view_num
#         for resolution_scale in cfg.resolution_scales:
#             print("Loading Training Cameras")
#             self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale)
#             print("Loading Test Cameras")
#             self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale)

#             self.world_view_transforms = []
#             camera_centers = []
#             center_rays = []
#             for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
#                 self.world_view_transforms.append(cur_cam.world_view_transform)
#                 camera_centers.append(cur_cam.camera_center)
#                 R = torch.tensor(cur_cam.R).float().cuda()
#                 T = torch.tensor(cur_cam.T).float().cuda()
#                 center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
#                 center_ray = center_ray@R.transpose(-1,-2)
#                 center_rays.append(center_ray)
#             self.world_view_transforms = torch.stack(self.world_view_transforms)
#             camera_centers = torch.stack(camera_centers, dim=0)
#             center_rays = torch.stack(center_rays, dim=0)
#             center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
#             diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy()
#             tmp = torch.sum(center_rays[:,None]*center_rays[None], dim=-1)
#             angles = torch.arccos(tmp)*180/3.14159
#             angles = angles.detach().cpu().numpy()
#             with open(os.path.join(self.model_path, "multi_view.json"), 'w') as file:
#                 for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
#                     print("查看cur_cam",cur_cam.id)
#                     print("查看cur_cam",cur_cam.image_name)
#                     sorted_indices = np.lexsort((angles[id], diss[id]))
#                     # sorted_indices = np.lexsort((diss[id], angles[id]))
#                     mask = (angles[id][sorted_indices] < cfg.multi_view_max_angle) & \
#                         (diss[id][sorted_indices] > cfg.multi_view_min_dis) & \
#                         (diss[id][sorted_indices] < cfg.multi_view_max_dis)
#                     sorted_indices = sorted_indices[mask]
#                     multi_view_num = min(self.multi_view_num, len(sorted_indices))
#                     json_d = {'ref_name' : cur_cam.image_name, 'nearest_name': []}
#                     for index in sorted_indices[:multi_view_num]:
#                         cur_cam.nearest_id.append(index)
#                         cur_cam.nearest_names.append(self.train_cameras[resolution_scale][index].image_name)
#                         json_d["nearest_name"].append(self.train_cameras[resolution_scale][index].image_name)
#                     json_str = json.dumps(json_d, separators=(',', ':'))
#                     file.write(json_str)
#                     file.write('\n')

import os
import random
import json
import torch
import numpy as np
from lib.utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from lib.config import cfg
from lib.datasets.base_readers import storePly, SceneInfo
from lib.datasets.colmap_readers import readColmapSceneInfo
from lib.datasets.blender_readers import readNerfSyntheticInfo
from lib.datasets.waymo_full_readers import readWaymoFullInfo

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Waymo": readWaymoFullInfo,
}

class Dataset():
    def __init__(self):
        self.cfg = cfg.data
        self.model_path = cfg.model_path
        self.source_path = cfg.source_path
        self.images = self.cfg.images

        self.train_cameras = {}
        self.test_cameras = {}

        dataset_type = cfg.data.get('type', "Colmap")
        assert dataset_type in sceneLoadTypeCallbacks.keys(), 'Could not recognize scene type!'
        
        scene_info: SceneInfo = sceneLoadTypeCallbacks[dataset_type](self.source_path, **cfg.data)

        if cfg.mode == 'train':
            print(f'Saving input pointcloud to {os.path.join(self.model_path, "input.ply")}')
            pcd = scene_info.point_cloud
            storePly(os.path.join(self.model_path, "input.ply"), pcd.points, pcd.colors)

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))

            print(f'Saving input camera to {os.path.join(self.model_path, "cameras.json")}')
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
       
        self.scene_info = scene_info
        
        if self.cfg.shuffle and cfg.mode == 'train':
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f"cameras_extent {self.cameras_extent}")

        self.multi_view_num = cfg.multi_view_num
        for resolution_scale in cfg.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale)

            self.world_view_transforms = []
            camera_centers = []
            center_rays = []


        with open(os.path.join(self.model_path, "multi_view_lt.json"), 'w') as file:
            for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                self.world_view_transforms.append(cur_cam.world_view_transform)
                camera_centers.append(cur_cam.camera_center)
                print(cur_cam.image_name)
                cur_cam_number = int(cur_cam.image_name.split('_')[0])
                suffix = cur_cam.image_name.split('_')[1]

                # 计算前一个和后一个相机的编号
                prev_cam_number = cur_cam_number - 1
                next_cam_number = cur_cam_number + 1

                # 构造前一个和后一个相机的名字
                prev_cam_name = f"{prev_cam_number:06d}_{suffix}"
                next_cam_name = f"{next_cam_number:06d}_{suffix}"
                cur_cam.nearest_names.append(prev_cam_name)
                cur_cam.nearest_names.append(next_cam_name)
                # 假设 self.train_cameras 是一个列表，resolution_scale 是一个索引
                # 假设 prev_cam_number 和 next_cam_number 是我们想要找到的相机编号

                # 遍历 self.train_cameras[resolution_scale] 列表
                for i, camera in enumerate(self.train_cameras[resolution_scale]):
                    # 提取当前相机的编号
                    current_cam_number = int(camera.image_name.split('_')[0])
    
                    # 检查当前相机编号是否与我们要找的编号匹配
                    if camera.image_name == prev_cam_name:
                        # 如果匹配，将索引添加到 cur_cam.nearest_id 中
                        cur_cam.nearest_id.append(i)
                    elif camera.image_name == next_cam_number:
                        # 如果匹配，也将索引添加到 cur_cam.nearest_id 中
                        cur_cam.nearest_id.append(i)
        
                json_d = {
                        'ref_name': cur_cam.image_name,
                        'nearest_name': cur_cam.nearest_names,
                        'nearest_id': cur_cam.nearest_id
                    }
                    
                json_str = json.dumps(json_d, separators=(',', ':'))
                file.write(json_str)
                file.write('\n')
                R = torch.tensor(cur_cam.R).float().cuda()
                T = torch.tensor(cur_cam.T).float().cuda()
                center_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
                center_ray = center_ray @ R.transpose(-1, -2)
                center_rays.append(center_ray)

                # Determine nearest neighbors based on custom rules