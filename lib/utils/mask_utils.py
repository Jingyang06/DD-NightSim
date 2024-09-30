def hist_mask(light_color):
    light_mask_high = (light_color <= 200)
    light_mask_low = (light_color >= 20)
    light_mask = light_mask_high * light_mask_low
    return light_mask



def auto_illu_mask(illu_k):
    # print(illu_k.shape)

    illu_k = illu_k.clamp(min=1e-4)
    # print(illu_k.shape)

    

    _, _, _ = illu_k.shape
    i_k = illu_k
    key_point = 0.05 * (i_k.median() - i_k.min())
    la = i_k.min() + key_point
    key_point = 0.1 * (i_k.max() - i_k.median())
    lb = i_k.max() - key_point
    if lb < 0.92:
        lb = 0.92
    else:
        lb = lb
    illu_mask_mid = (i_k > la) * (i_k < lb)
    illu_mask_low = (1 / (1 + (i_k < la) * (i_k - la) * (i_k - la) * (5 ** 2))) * (i_k < la)
    illu_mask_high = (1 / (1 + (i_k > lb) * (i_k - lb) * (i_k - lb) * (5 ** 2))) * (i_k > lb)
    illu_mask = illu_mask_mid + illu_mask_low + illu_mask_high
    illu_mask = illu_mask.detach()
    k = 1 / (lb - la)
    # k = torch.stack(scales).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return illu_mask, k
    