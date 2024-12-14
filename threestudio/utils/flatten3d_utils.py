from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from pathlib import Path
from PIL import Image
import torch
import yaml
import math

from gaussiansplatting.utils.graphics_utils import fov2focal, depth_to_3d
import torchvision.transforms as T
from torchvision.io import read_video,write_video
import os
import random
import numpy as np
from torchvision.io import write_video

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def resize_bool_tensor(bool_tensor, size):
    """
    Resizes a boolean tensor to a new size using nearest neighbor interpolation.
    """
    # Convert boolean tensor to float
    H_new, W_new = size
    tensor_float = bool_tensor.float()

    # Resize using nearest interpolation
    resized_float = torch.nn.functional.interpolate(tensor_float, size=(H_new, W_new), mode='nearest')

    # Convert back to boolean
    resized_bool = resized_float > 0.5
    return resized_bool

def point_to_line_dist(points, lines):
    """
    Calculate the distance from points to lines in 2D.
    points: Nx3
    lines: Mx3

    return distance: NxM
    """
    numerator = torch.abs(lines @ points.T)
    denominator = torch.linalg.norm(lines[:,:2], dim=1, keepdim=True)
    return numerator / denominator

def save_video_frames(video_path, img_size=(512,512)):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith('.mov'):
        video = T.functional.rotate(video, -90)
    video_name = Path(video_path).stem
    os.makedirs(f'data/{video_name}', exist_ok=True)
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize((img_size),  resample=Image.Resampling.LANCZOS)
        image_resized.save(f'data/{video_name}/{ind}.png')

def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
        
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def load_imgs(data_path, n_frames, device='cuda', pil=False):
    imgs = []
    pils = []
    for i in range(n_frames):
        img_path = os.path.join(data_path, "%05d.jpg" % i)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, "%05d.png" % i)
        img_pil = Image.open(img_path)
        pils.append(img_pil)
        img = T.ToTensor()(img_pil).unsqueeze(0)
        imgs.append(img)
    if pil:
        return torch.cat(imgs).to(device), pils
    return torch.cat(imgs).to(device)


def save_video(raw_frames, save_path, fps=10):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }

    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)
    

def compute_depth_correspondence(cam1, cam2, depth1, depth2, current_H=64, current_W=64):
    """compute depth map error of cam2 with respect to cam1

    Args:
        cam1 (_type_): _description_
        cam2 (_type_): _description_
        depth1 (_type_): _description_
        depth2 (_type_): _description_
        current_H (int, optional): _description_. Defaults to 64.
        current_W (int, optional): _description_. Defaults to 64.
    """
    batch_size, _, DH, DW = depth1.shape
    intermediate_dtype = torch.float32
    dtype = depth1.dtype
    device = depth1.device
    # depth1 = F.interpolate(depth1, (current_H, current_W), mode="bilinear", align_corners=False)
    # depth2 = F.interpolate(depth2, (current_H, current_W), mode="bilinear", align_corners=False)
    
    # cam_1_view2world = torch.inverse(cam1.world_view_transform)[:, :].T.unsqueeze(0).to(intermediate_dtype)
    cam_2_view2world = torch.inverse(cam2.world_view_transform)[:, :].T.unsqueeze(0).to(intermediate_dtype)
    
    K1 = torch.tensor(
        [[[fov2focal(cam1.FoVx, DW), 0, DW / 2], [0, fov2focal(cam1.FoVy, DH), DH / 2], [0, 0, 1]]],
        dtype=intermediate_dtype,
        device=depth1.device
    )
    K2 = torch.tensor(
        [[[fov2focal(cam2.FoVx, DW), 0, DW / 2], [0, fov2focal(cam2.FoVy, DH), DH / 2], [0, 0, 1]]],
        dtype=intermediate_dtype,
        device=depth2.device
    )
    
    cam_1_pixel = K1 @ cam1.world_view_transform[:, [0,1,2]].T
    
    homogeneous_filler = torch.ones_like(depth1.to(intermediate_dtype)) # shape B 1 H W
    # points1 = torch.cat([depth_to_3d(depth1, K1), homogeneous_filler], dim=1) # shape B 4 H W
    points2 = torch.cat([depth_to_3d(depth2.to(intermediate_dtype), K2), homogeneous_filler], dim=1)
    # points1 = torch.bmm(cam_1_view2world, points1.view(batch_size, 4, DH * DW)).view(batch_size, 4, DH, DW)
    points2 = torch.bmm(cam_2_view2world, points2.view(batch_size, 4, DH * DW)).view(batch_size, 4, DH, DW).to(intermediate_dtype)
    points2_on_1 = torch.bmm(cam_1_pixel, points2.view(batch_size, 4, DH * DW)).view(batch_size, 3, DH, DW).to(intermediate_dtype)
    points2_on_1 = (points2_on_1[:, :2, :, :] / points2_on_1[:, [2], :, :]).view(2, DH, DW) # 2 H W, disposing homogeneous item
    
    valid_mask = torch.logical_and(
        torch.logical_and(points2_on_1[0] >= 0, points2_on_1[0] < DW),
        torch.logical_and(points2_on_1[1] >= 0, points2_on_1[1] < DH)
    )[None, :, :]
    
    points2_binned = torch.round(points2_on_1.permute(1, 2, 0) - 0.5).view(DH * DW, 2).long() # H*W 2 (x, y)
    bin_mask = torch.logical_and(
        torch.logical_and(points2_on_1[0] >= 0, points2_on_1[0] < DW),
        torch.logical_and(points2_on_1[1] >= 0, points2_on_1[1] < DH)
    ).view(-1)
    points2_binned = points2_binned[bin_mask]
    points2_binned = points2_binned[:, 1] * DW + points2_binned[:, 0] # H*W
    
    depth2_flat = depth2.view(DH * DW)
    
    valid_depth = torch.zeros_like(depth2_flat)
    valid_depth.scatter_reduce_(0, points2_binned, depth2_flat[bin_mask.view(-1)], "amin", include_self=False)
    max_depth = valid_depth.max()
    
    valid_depth_mask = (depth2_flat < max_depth).view(1, DH, DW)
    
    # mask on map 2 to indicate valid pixels
    valid_mask = torch.logical_and(valid_mask, valid_depth_mask).view(DH * DW)
    
    x,y = torch.meshgrid([torch.arange(DW), torch.arange(DH)], indexing="xy")
    points2_on_2 = 0.5 + torch.stack([x, y], dim=2).view(DH, DW, 2).to(device)
    
    points2_on_1_scaled = points2_on_1 * torch.tensor([current_W / DW, current_H / DH]).view(2, 1, 1).to(device) # C H W
    points2_on_2_scaled = points2_on_2 * torch.tensor([current_W / DW, current_H / DH]).view(1, 1, 2).to(device) # H W C
    points2_on_2_scaled_pixel = torch.round(points2_on_2_scaled - 0.5).long() # [0, W-1] [0, H-1] range
    points2_on_2_scaled_pixel = points2_on_2_scaled_pixel[:, :, 1] * current_W + points2_on_2_scaled_pixel[:, :, 0] # flat coordinates
    
    # caused by float16 and float32 precision
    assert torch.all(points2_on_2_scaled_pixel.view(1, DH * DW)[:, valid_mask] >= 0) and torch.all(points2_on_2_scaled_pixel.view(1, DH * DW)[:, valid_mask] < current_H * current_W)
    
    points2_on_1_collected = torch.ones([2, current_H * current_W], dtype=intermediate_dtype, device=device) * -1
    points2_on_1_collected.scatter_reduce_(1, 
        points2_on_2_scaled_pixel.view(1, DH * DW).repeat(2, 1)[:, valid_mask], 
        points2_on_1_scaled.view(2, DH * DW)[:, valid_mask], "mean", include_self=False)
    
    points2_on_1_collected = points2_on_1_collected.view(2, current_H, current_W)
    valid_mask = (points2_on_1_collected[0] != -1).view(1, current_H, current_W)
    
    norm_factor = torch.tensor([current_W, current_H]).view(1, 2, 1, 1).to(device)
    points2_on_1_norm = (points2_on_1_collected / norm_factor).view(2, current_W, current_H).to(dtype) # C H W output, [0, 1] range
    points2_on_1_norm = 2 * (points2_on_1_norm - 0.5) # [-1, 1] range
    
    return points2_on_1_norm, valid_mask # C H W output

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_depth_correspondence(diffusion_model, depth_correspondence, depth_valid_mask):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "depth_correspondence", depth_correspondence)
            setattr(module, "depth_valid_mask", depth_valid_mask)

def register_cams(diffusion_model, cams, pivot_this_batch, key_cams):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "cams", cams)
            setattr(module, "pivot_this_batch", pivot_this_batch)
            setattr(module, "key_cams", key_cams)

def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)
            
def unregister_pivotal_data(diffusion_model):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            module.cleanup_cache()
    torch.cuda.empty_cache()
            
def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_t(diffusion_model, t):

    for _, module in diffusion_model.named_modules():
    # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "t", t)
            
def register_corre_attn_strength(diffusion_model, corre_attn_strength):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.corre_attn_strength = corre_attn_strength
            
def register_low_vram(diffusion_model, low_vram):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.low_vram = low_vram

def register_normal_attention(model):
    # def sa_forward(self):
    #     to_out = self.to_out
    #     if type(to_out) is torch.nn.modules.container.ModuleList:
    #         to_out = self.to_out[0]
    #     else:
    #         to_out = self.to_out
    #     def forward(x, encoder_hidden_states=None, attention_mask=None):
    #         # assert encoder_hidden_states is None 
    #         batch_size, sequence_length, dim = x.shape
    #         h = self.heads
    #         is_cross = encoder_hidden_states is not None
    #         encoder_hidden_states = encoder_hidden_states if is_cross else x
    #         q = self.to_q(x)
    #         k = self.to_k(encoder_hidden_states)
    #         v = self.to_v(encoder_hidden_states)

    #         if self.group_norm is not None:
    #             hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    #         query = self.head_to_batch_dim(q)
    #         key = self.head_to_batch_dim(k)
    #         value = self.head_to_batch_dim(v)

    #         attention_probs = self.get_attention_scores(query, key)
    #         hidden_states = torch.bmm(attention_probs, value)
    #         out = self.batch_to_head_dim(hidden_states)

    #         return to_out(out)

    #     return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            # module.attn1.normal_attn = sa_forward(module.attn1)
            module.use_normal_attn = True

def register_normal_attn_flag(diffusion_model, use_normal_attn):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "use_normal_attn", use_normal_attn)

def register_extended_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        def extended_forward(x, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            assert encoder_hidden_states is None 
            assert attention_mask is None
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x

            skip_map = cross_attention_kwargs.get("skip_map", {})
            
            skippable_to_q = nn.Identity() if skip_map.get("q", False) else self.to_q
            skippable_to_k = nn.Identity() if skip_map.get("k", False) else self.to_k
            skippable_to_v = nn.Identity() if skip_map.get("v", False) else self.to_v
            skippable_to_out = nn.Identity() if skip_map.get("out", False) else to_out
            
            q = skippable_to_q(x)
            k = skippable_to_k(encoder_hidden_states)
            v = skippable_to_v(encoder_hidden_states)
            
            k_text = k[:n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_image = k[n_frames: 2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_uncond = k[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            v_text = v[:n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_image = v[n_frames:2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_uncond = v[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_text = self.head_to_batch_dim(q[:n_frames])
            q_image = self.head_to_batch_dim(q[n_frames: 2*n_frames])
            q_uncond = self.head_to_batch_dim(q[2 * n_frames:])

            k_text = self.head_to_batch_dim(k_text)
            k_image = self.head_to_batch_dim(k_image)
            k_uncond = self.head_to_batch_dim(k_uncond)

            
            v_text = self.head_to_batch_dim(v_text)
            v_image = self.head_to_batch_dim(v_image)
            v_uncond = self.head_to_batch_dim(v_uncond)

            out_text = []
            out_image = []
            out_uncond = []

            q_text = q_text.view(n_frames, h, sequence_length, dim // h)
            k_text = k_text.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_text = v_text.view(n_frames, h, sequence_length * n_frames, dim // h)

            q_image = q_image.view(n_frames, h, sequence_length, dim // h)
            k_image = k_image.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_image = v_image.view(n_frames, h, sequence_length * n_frames, dim // h)

            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)

            for j in range(h):
                sim_text = torch.bmm(q_text[:, j], k_text[:, j].transpose(-1, -2)) * self.scale
                sim_image = torch.bmm(q_image[:, j], k_image[:, j].transpose(-1, -2)) * self.scale
                sim_uncond = torch.bmm(q_uncond[:, j], k_uncond[:, j].transpose(-1, -2)) * self.scale
                
                out_text.append(torch.bmm(sim_text.softmax(dim=-1), v_text[:, j]))
                out_image.append(torch.bmm(sim_image.softmax(dim=-1), v_image[:, j]))
                out_uncond.append(torch.bmm(sim_uncond.softmax(dim=-1), v_uncond[:, j]))

            out_text = torch.cat(out_text, dim=0).view(h, n_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_image = torch.cat(out_image, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_uncond = torch.cat(out_uncond, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)

            out = torch.cat([out_text, out_image, out_uncond], dim=0)
            out = self.batch_to_head_dim(out)

            out = skippable_to_out(out)
            
            return out
        
        def skippable_normal_forward(x, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            # assert encoder_hidden_states is None 
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            
            skip_map = cross_attention_kwargs.get("skip_map", {})
            key_override = cross_attention_kwargs.get("key_override", None)
            
            skippable_to_q = nn.Identity() if skip_map.get("q", False) else self.to_q
            skippable_to_k = nn.Identity() if skip_map.get("k", False) else self.to_k
            skippable_to_v = nn.Identity() if skip_map.get("v", False) else self.to_v
            skippable_to_out = nn.Identity() if skip_map.get("out", False) else to_out
            
            if key_override is not None:
                q = skippable_to_q(x)
                k = skippable_to_k(key_override)
                v = skippable_to_v(encoder_hidden_states)
            else:
                q = skippable_to_q(x)
                k = skippable_to_k(encoder_hidden_states)
                v = skippable_to_v(encoder_hidden_states)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.head_to_batch_dim(q)
            key = self.head_to_batch_dim(k)
            value = self.head_to_batch_dim(v)
            
            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            # attention_probs = self.get_attention_scores(query, key)
            # hidden_states = torch.bmm(attention_probs, value)
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=self.scale)
            out = self.batch_to_head_dim(hidden_states)

            out = skippable_to_out(out)
                
            return out
        
        def forward(x, encoder_hidden_states=None, attention_mask=None, use_normal_attn=False, **cross_attention_kwargs):
            skip_map = cross_attention_kwargs.get("skip_map", {})
            if use_normal_attn:
                if len(skip_map) == 0: return self.orig_forward(x, encoder_hidden_states, attention_mask)
                else: return skippable_normal_forward(x, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
            else:
                return extended_forward(x, encoder_hidden_states, attention_mask, **cross_attention_kwargs)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module.attn1,"orig_forward", module.attn1.forward)
            module.attn1.forward = sa_forward(module.attn1)


def compute_camera_distance(cams, key_cams):
    cam_centers = [cam.camera_center for cam in cams]
    key_cam_centers = [cam.camera_center for cam in key_cams] 
    cam_centers = torch.stack(cam_centers).cuda()
    key_cam_centers = torch.stack(key_cam_centers).cuda()
    cam_distance = torch.cdist(cam_centers, key_cam_centers)

    return cam_distance   

def make_flatten3d_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class Flatten3DBlock(block_class):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            
            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 3
            hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
        
            norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)
            if self.pivotal_pass:
                self.pivot_hidden_states = norm_hidden_states
            if not self.use_normal_attn:
                if self.pivotal_pass:
                    self.pivot_hidden_states = norm_hidden_states
                else:
                    batch_idxs = [self.batch_idx]
                    if self.batch_idx > 0:
                        batch_idxs.append(self.batch_idx - 1)
                    cam_distance = compute_camera_distance(self.cams, self.key_cams)
                    cam_distance_min = cam_distance.sort(dim=-1)
                    closest_cam = cam_distance_min[1][:,:len(batch_idxs)]
                    closest_cam_pivot_hidden_states = self.pivot_hidden_states[:, closest_cam].view(3, n_frames, len(batch_idxs) * sequence_length, dim)
                    
                    key_frame_count = len(self.key_cams)
                    key_cam_selector = torch.arange(key_frame_count).view(1, -1).repeat(n_frames, 1).cuda()
                    # shape (frames, len(keycams), 2, DH, DW)
                    depth_correspondence = self.depth_correspondence[sequence_length].gather(dim=1, index=key_cam_selector[:, :, None, None, None].expand(-1, -1, *self.depth_correspondence[sequence_length].shape[2:]))
                    depth_valid_mask = self.depth_valid_mask[sequence_length].gather(dim=1, index=key_cam_selector[:, :, None, None, None].expand(-1, -1, *self.depth_valid_mask[sequence_length].shape[2:]))
            
            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if type(self.attn1.to_out) is torch.nn.modules.container.ModuleList:
                to_out = self.attn1.to_out[0]
            else:
                to_out = self.attn1.to_out
            if self.use_normal_attn:
                # print("use normal attn")
                attn_output = self.attn1(
                        norm_hidden_states.view(batch_size, sequence_length, dim),
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        use_normal_attn=self.use_normal_attn,
                        **cross_attention_kwargs,
                    )         
            else:
                # print("use extend attn")
                if self.pivotal_pass:
                    # norm_hidden_states.shape = 3, n_frames * seq_len, dim
                    attn_output = self.attn1(
                            norm_hidden_states.view(batch_size, sequence_length, dim),
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            use_normal_attn=self.use_normal_attn,
                            **cross_attention_kwargs,
                        )
                    # 3, n_frames * seq_len, dim - > 3 * n_frames, seq_len, dim
                    self.kf_attn_output = attn_output

                else:
                    _, _, _, DH, DW = depth_correspondence.shape
                    depth_grid = depth_correspondence.permute(0, 1, 3, 4, 2) # frames, keycams, DH, DW, 2
                    # kf attn output, not passed to_out
                    pivot_hidden_states = self.pivot_hidden_states.view(3, key_frame_count, DH, DW, dim).permute(0, 1, 4, 2, 3)
                    
                    sampled_pivot_hidden_state = F.grid_sample(
                        pivot_hidden_states.repeat(1, n_frames, 1, 1, 1).view(3 * n_frames * key_frame_count, dim, DH, DW),
                        depth_grid.repeat(3, 1, 1, 1, 1).view(3 * n_frames * key_frame_count, DH, DW, 2),
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=False
                    ).view(3 * n_frames, key_frame_count, dim, DH * DW).permute(0, 1, 3, 2).reshape(3, n_frames, key_frame_count * sequence_length, dim)
                    
                    attn_mask = depth_valid_mask.view(1, n_frames, 1, key_frame_count, sequence_length)
                    filter_flatten = attn_mask.squeeze().sum(dim=-2, keepdim=True) > 0
                    attn_mask = torch.cat([~filter_flatten[None, :, None, :, :], attn_mask], dim=-2).repeat(3, 1, 1, 1, 1)
                    attn_mask = attn_mask.view(3, n_frames, 1, (key_frame_count + 1) * sequence_length)
                    
                    kf_attn_output = self.kf_attn_output.view(3, key_frame_count, DH, DW, dim).permute(0, 1, 4, 2, 3)
                    sampled_kf_attn_output = F.grid_sample(
                        kf_attn_output.repeat(1, n_frames, 1, 1, 1).view(3 * n_frames * key_frame_count, dim, DH, DW),
                        depth_grid.repeat(3, 1, 1, 1, 1).view(3 * n_frames * key_frame_count, DH, DW, 2),
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=False
                    ).view(3 * n_frames, key_frame_count, dim, DH * DW).permute(0, 3, 1, 2).reshape(3 * n_frames * sequence_length, key_frame_count, dim)
                    flatten_attn_mask = depth_valid_mask.view(1, n_frames, 1, key_frame_count, sequence_length).permute(0, 1, 4, 2, 3).repeat(3, 1, 1, 1, 1)
                    flatten_attn_mask = flatten_attn_mask.view(batch_size * sequence_length, 1, key_frame_count)
                    flatten_attn_filter = flatten_attn_mask.view(batch_size * sequence_length, key_frame_count).sum(dim=-1) > 0
                    
                    # do FLATTEN attention (IS shape correct?)
                    # sampled_attn_outputs = sampled_attn_outputs.permute(0, 2, 1, 3).reshape(3 * n_frames * sequence_length, key_frame_count, dim)
                    # attn_mask = attn_mask.permute(0, 2, 1).reshape(3 * n_frames * sequence_length, 1, key_frame_count) # mask on keys                
                    
                    # cross attention towards key frames
                    if self.low_vram:
                        attn_output = []
                        for i in range(norm_hidden_states.shape[1]):
                            attn_output.append(
                                self.attn1(
                                    norm_hidden_states[:, i],
                                    torch.cat([
                                        norm_hidden_states[:, i], 
                                        sampled_pivot_hidden_state[:, i]
                                        ], dim=-2), # TODO maybe concat with original KV for better performance?
                                    attention_mask=attn_mask[:, i],
                                    use_normal_attn=True,
                                    **cross_attention_kwargs
                                )
                            )
                        attn_output = torch.stack(attn_output, dim=1).reshape(batch_size, sequence_length, dim).half()  
                    else:
                        attn_output = self.attn1(
                            norm_hidden_states.view(batch_size, sequence_length, dim),
                            torch.cat([
                                norm_hidden_states.view(batch_size, sequence_length, dim), 
                                sampled_pivot_hidden_state.view(batch_size, key_frame_count * sequence_length, dim)
                                ], dim=1).view(batch_size, -1, dim), # TODO maybe concat with original KV for better performance?
                            attention_mask=attn_mask.view(batch_size, 1, -1),
                            use_normal_attn=True,
                            **cross_attention_kwargs
                        ).view(batch_size * sequence_length, 1, dim).half() 
                        
                    
                    if sequence_length == np.max(list(self.depth_correspondence.keys())) and self.corre_attn_strength > 0:
                        corre_attn_output = self.attn1(
                            norm_hidden_states.view(batch_size * sequence_length, 1, dim)[flatten_attn_filter],
                            encoder_hidden_states=sampled_kf_attn_output.view(batch_size * sequence_length, key_frame_count, dim)[flatten_attn_filter],
                            attention_mask=flatten_attn_mask.view(batch_size * sequence_length, 1, key_frame_count)[flatten_attn_filter],
                            key_override=sampled_pivot_hidden_state.
                                view(batch_size, key_frame_count, sequence_length, dim).
                                permute(0, 2, 1, 3).
                                reshape(batch_size * sequence_length, key_frame_count, dim)[flatten_attn_filter],
                            skip_map={"q":True, "k":True, "v":True, "out":True},
                            use_normal_attn=True,
                            **cross_attention_kwargs
                        )
                        
                        attn_output[flatten_attn_filter] = corre_attn_output * self.corre_attn_strength + attn_output[flatten_attn_filter] * (1-self.corre_attn_strength)
                    
                    attn_output = attn_output.view(batch_size, sequence_length, dim).half() 

                        
            if self.use_ada_layer_norm_zero:
                self.n = gate_msa.unsqueeze(1) * attn_output               
            
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            hidden_states = attn_output + hidden_states
            hidden_states = hidden_states.to(self.norm2.weight.dtype)
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]


            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states
        
        def cleanup_cache(self):
            if hasattr(self, "pivot_hidden_states"):
                del self.pivot_hidden_states
            if hasattr(self, "kf_attn_output"):
                del self.kf_attn_output

    return Flatten3DBlock

