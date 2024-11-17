#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch, torchvision
from random import randint
# from Three_dgs_kitti.utils.loss_utils import l2_loss, ssim, l1_loss, ScaleAndShiftInvariantLoss
# from Three_dgs_kitti.gaussian_renderer import render, network_gui, render_dyn, render_all, return_gaussians_boxes_and_box2worlds
# from Three_dgs_kitti.scene.cameras import augmentCamera
from PIL import Image
import sys
# from Three_dgs_kitti.scene import Scene, GaussianModel, GaussianBoxModel
# from Three_dgs_kitti.utils.general_utils import safe_state, Normal2Torch
import uuid
from tqdm import tqdm
# from Three_dgs_kitti.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
# from Three_dgs_kitti.arguments import ModelParams, PipelineParams, OptimizationParams, KITTI360DataParams, BoxModelParams, SDRegularizationParams
from kitti360scripts.helpers import labels as kittilabels
import random
import numpy as np
# from Three_dgs_kitti.utils.graphics_utils import normal_to_rot, cam_normal_to_world_normal, standardize_quaternion, matrix_to_quaternion, quaternion_to_matrix
# from Three_dgs_kitti.loss import loss_normal_guidance
import wandb

# from Three_dgs_kitti.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from Three_dgs_kitti.utils.loss_utils import l2_loss
import copy

from Test_render.cameras import make_camera_like_input_camera
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm
import math


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
torch.autograd.set_detect_anomaly(True)

def seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

@torch.no_grad()
def initialize_gaussians_with_normals(gaussians, scene, pipe, background):
    print(__name__)    
    
    # validataion
    ptr_gs_rot = gaussians._rotation.data_ptr
    ptr_gs_scale = gaussians._scaling.data_ptr
    # snu_checkpoint = f"submodules/surface_normal_uncertainty/checkpoints/{snu_args.pretrained}.pt"
    # print('loading snu checkpoint... {}'.format(snu_checkpoint))
    # print('loading snu checkpoint... / done')

    viewpoint_stack = scene.getTrainCameras()
    n_cameras = len(viewpoint_stack)
    # quaternion_new = torch.zeros_like(gaussians._rotation)
    quaternion_new = copy.deepcopy(gaussians._rotation)
    # quaternion_new_cnt = torch.ones((gaussians._rotation.shape[0]), device=quaternion_new.device) 
    # visibility_mark_whole = torch.zeros((gaussians.get_xyz.shape[0]))>0

    with tqdm(range(n_cameras)) as pbar:
        pbar.set_description("initialize with normal prediction")
        for i in pbar:        
            # estimate rotations from normals ---------------------------
            viewpoint_cam = viewpoint_stack[i]
            norm_pred = viewpoint_cam.original_normal
            _, H, W = norm_pred.shape

            norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)
            rot_from_norm = normal_to_rot(norm_pred_world.permute(1,2,0).reshape(-1,3))  # world2newworld      # n_pix, 3, 3  
            # rot_from_norm = rot_from_norm @ torch.from_numpy(viewpoint_cam.R[None, :, :]).to(norm_pred.device).float() # n_pix, 3, 3  
            #         
            # rot_from_norm = normal_to_rot(norm_pred_cam.permute(1,2,0).reshape(-1,3)) # new cam2world        
            quat_from_norm = matrix_to_quaternion(rot_from_norm)
            quat_from_norm = standardize_quaternion(quat_from_norm)
            # for later use
            

            # match 3d and normal -----------------------
            # all_bboxes = scene.getTrainBboxes() # This includes all existing dynamic objects' bboxes.
            
            # Render
            # frame = viewpoint_cam.frame
            # this_frame_includes_objects = (frame in all_bboxes.keys())       
            
            # # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
            # screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
            # try:
            #     screenspace_points.retain_grad()
            # except:
            #     pass
            scaling_modifier = 1.0
            # Set up rasterization configuration
            tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_cam.image_height),
                image_width=int(viewpoint_cam.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_cam.world_view_transform,
                projmatrix=viewpoint_cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=viewpoint_cam.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            visibility_mark =rasterizer.markVisible(gaussians.get_xyz)
            # visibility_mark_whole[visibility_mark] = True
                    
            # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # visibility_filter = render_pkg["visibility_filter"]
            # visibility_mark = render_pkg["visibility_mark"]
            # visible_xyz = torch.concat( (gaussians.get_xyz[visibility_filter], torch.ones((visibility_filter.sum(),1), device=gaussians.get_xyz.device)), dim=-1).unsqueeze(dim=-1)
            visible_xyz = (gaussians.get_xyz[visibility_mark]).unsqueeze(dim=-1)
            R = torch.from_numpy(viewpoint_cam.R).transpose(-1, -2).to(device=visible_xyz.device).type_as(visible_xyz) # world 2 cam
            T = torch.from_numpy(viewpoint_cam.T).to(device=visible_xyz.device).type_as(visible_xyz) # world 2 cam
            K = torch.from_numpy(viewpoint_cam.K).to(device=visible_xyz.device).type_as(visible_xyz) 
            visible_xyz_cam = ((R @ visible_xyz) + T[None, :, None])
            pix = (K @ visible_xyz_cam).squeeze()
            pix /= pix[:, -1:]

            # -1 ~ 1        
            pix[:, 0] = (pix[:, 0]*2 - W)/W
            pix[:, 1] = (pix[:, 1]*2 - H)/H
            
            quat_init = torch.nn.functional.grid_sample(quat_from_norm.permute(1,0).reshape(1, -1, H, W), pix[None, None, :, :-1], mode='nearest', align_corners=True)
            quat_init = standardize_quaternion(quat_init)
            
            # quaternion_new[visibility_mark] += (quat_init.squeeze().permute(1,0))
            # quaternion_new_cnt[visibility_mark] += 1
            mask_zero_quat = (quat_init.abs().squeeze().sum(dim=-2)< 1e-9)
            # if i==128: #(quaternion_new[visibility_mark])[23125].isinf().any():
            #     print('debug inf')
            quaternion_new[visibility_mark] = (quat_init.squeeze().permute(1,0)) * ~mask_zero_quat[:, None] + quaternion_new[visibility_mark] * mask_zero_quat[:, None] # exclude zeros
            # to_add = (quat_init.squeeze().permute(1,0)) * ~mask_zero_quat[:, None] + quaternion_new[visibility_mark] / quaternion_new_cnt[visibility_mark][:, None]* mask_zero_quat[:, None] # exclude zeros
            # if torch.isnan(to_add).any() or  torch.isinf(to_add).any():
            #     print('debug')
            # quaternion_new[visibility_mark] += to_add
            # if torch.isnan(quaternion_new[visibility_mark]).any() or torch.isinf(quaternion_new[visibility_mark]).any():
            #     print('debug')
            # quaternion_new_cnt[visibility_mark] +=  1*(~mask_zero_quat)

            # quaternion_to_matrix(gaussians.get_rotation[visibility_filter])
            # gaussians.get_xyz[visibility_filter].shape
    # quaternion_new /= quaternion_new_cnt.unsqueeze(dim=-1)
    # if torch.isnan(quaternion_new[visibility_mark]).any() or torch.isinf(quaternion_new[visibility_mark]).any() :
    #     print('debug2')
    # gaussians._rotation[visibility_mark_whole] = quaternion_new[visibility_mark_whole]    
    # gaussians._rotation.data = copy.deepcopy(quaternion_new / torch.linalg.norm(quaternion_new, axis=-1, keepdims=True))
    quaternion_new = standardize_quaternion(quaternion_new)
    quaternion_new /= (torch.linalg.norm(quaternion_new, axis=-1, keepdims=True) + 1e-9)
    # if torch.isnan(quaternion_new[visibility_mark]).any():
    #     print('debug3')
    # gaussians._rotation.data[: ,0] = 1.4142 * 0.3
    # gaussians._rotation.data[: ,1] = 1.4142 *0.4
    # gaussians._rotation.data[: ,2] = 1.4142 *0.5
    # gaussians._rotation.data[: ,3] = 0.0
    gaussians._rotation.data[: ,0] = copy.deepcopy(quaternion_new[:, 0].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,1] = copy.deepcopy(quaternion_new[:, 1].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,2] = copy.deepcopy(quaternion_new[:, 2].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,3] = copy.deepcopy(quaternion_new[:, 3].type_as(gaussians._rotation))
    
    # gaussians._rotation = copy.deepcopy(quaternion_new.type_as(gaussians._rotation)) # why not?
    # gaussians._rotation.data = quaternion_multiply( gaussians._rotation, quaternion_new)
    # self.rotations = rotations_from_quats / np.linalg.norm(rotations_from_quats, axis=-1, keepdims=True)

    
    # gaussians._scaling[:, 0] = copy.deepcopy(torch.log(torch.tensor([1e-9]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))

    assert gaussians._rotation.data_ptr == ptr_gs_rot
    assert gaussians._scaling.data_ptr == ptr_gs_scale

    # gaussians._scaling = (torch.ones_like(gaussians._scaling) * torch.tensor([[1e-9, 100, 100]], device=gaussians._scaling.device)).clone()
    # temp_scaling = torch.ones_like(gaussians._scaling) * torch.tensor([[1e-9, 100, 100]], device=gaussians._scaling.device)
    # gaussians._scaling = torch.nn.Parameter(temp_scaling.requires_grad_(True))
    # gaussians._rotation = torch.nn.Parameter(quaternion_new.requires_grad_(True))
    # gaussians._opacity = torch.nn.Parameter(opacities.requires_grad_(True))
    return gaussians

def sort_quaternion_candidates(quaternion_full):
    n_pnt_full = quaternion_full.shape[0]
    n_pnt_batch = 512

    # with tqdm(range(math.ceil(n_pnt_full / 512))) as pbar:
    #     pbar.set_description("refresh_quaternion_new")
    #     for i in pbar:
    for i in range(math.ceil(n_pnt_full / 512)):
        quaternion_full_subset = quaternion_full[i*n_pnt_batch: min((i+1)*n_pnt_batch, n_pnt_full)]
        quaternion_sim = quaternion_full_subset@quaternion_full_subset.permute(0,2,1)
        idx_best_normal = torch.argsort(quaternion_sim.sum(dim=2), dim=1, descending=True)
        quaternion_full[i*n_pnt_batch: min((i+1)*n_pnt_batch, n_pnt_full)] = torch.gather(quaternion_full_subset, index=idx_best_normal[:, :, None].repeat(1,1,4), dim=1)
    return quaternion_full
def refresh_quaternion_new(quaternion_new, idx_accum_quat):
    max_memory = idx_accum_quat.max().item()
    new_memory = int(max_memory *0.7)
    
    # check mem full
    # mask_full = (idx_accum_quat == idx_accum_quat.max())
    mask_full = (idx_accum_quat >=new_memory)
    # quaternion_new.shape
    # idx_accum_quat.shape

    quaternion_full = quaternion_new[mask_full]
    quaternion_full = sort_quaternion_candidates(quaternion_full)
    quaternion_new[mask_full] = quaternion_full
    idx_accum_quat[mask_full] = new_memory

    # temp = quaternion_new[mask_full]@quaternion_new[mask_full].permute(0,2,1)
    # idx_best_normal = torch.argsort(temp.sum(dim=2), dim=1, descending=True)
    # quaternion_new[mask_full] = torch.gather(quaternion_new[mask_full], index=idx_best_normal[:, :, None].repeat(1,1,4), dim=1)
    # idx_accum_quat[mask_full] = new_memory

    return quaternion_new, idx_accum_quat

def best_quaternion_new(quaternion_new, idx_accum_quat):
    max_memory = quaternion_new.shape[1]
    for n_accum in idx_accum_quat.unique():
        mask = (idx_accum_quat == n_accum)[:, None].repeat(1, max_memory)
        mask[:, n_accum:] = False
        quaternion_accum = quaternion_new[mask].reshape(-1, n_accum, 4)
        quaternion_accum = sort_quaternion_candidates(quaternion_accum)
        quaternion_new[mask] = quaternion_accum.reshape(-1,4)        
    
    return quaternion_new[:, 0, :]

@torch.no_grad()
def initialize_gaussians_with_window_normals(gaussians, scene, pipe, background):
    print(__name__)    
    
    # validataion
    ptr_gs_rot = gaussians._rotation.data_ptr
    ptr_gs_scale = gaussians._scaling.data_ptr
    
    viewpoint_stack = scene.getTrainCameras()
    n_cameras = len(viewpoint_stack)

    max_memory = 100
    quaternion_new = copy.deepcopy(gaussians._rotation).reshape(-1,1,4).repeat(1,max_memory,1) # consider max 100 normal at once
    n_pnt = quaternion_new.shape[0]
    idx_accum_quat = torch.zeros((n_pnt)).to(quaternion_new.device).to(torch.int64)

    with tqdm(range(n_cameras)) as pbar:
        pbar.set_description("initialize with normal prediction")
        for i in pbar:        
            # estimate rotations from normals ---------------------------
            viewpoint_cam = viewpoint_stack[i]
            norm_pred = viewpoint_cam.original_normal
            _, H, W = norm_pred.shape

            norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)
            rot_from_norm = normal_to_rot(norm_pred_world.permute(1,2,0).reshape(-1,3))  # world2newworld      # n_pix, 3, 3  
             
            quat_from_norm = matrix_to_quaternion(rot_from_norm)
            quat_from_norm = standardize_quaternion(quat_from_norm)
            
            # match 2d / 3d normal -----------------------            
            scaling_modifier = 1.0
            # Set up rasterization configuration
            tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_cam.image_height),
                image_width=int(viewpoint_cam.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_cam.world_view_transform,
                projmatrix=viewpoint_cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=viewpoint_cam.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            visibility_mark =rasterizer.markVisible(gaussians.get_xyz) # visibility_mark.shape: torch.Size([2370245])
                    
            visible_xyz = (gaussians.get_xyz[visibility_mark]).unsqueeze(dim=-1)
            R = torch.from_numpy(viewpoint_cam.R).transpose(-1, -2).to(device=visible_xyz.device).type_as(visible_xyz) # world 2 cam
            T = torch.from_numpy(viewpoint_cam.T).to(device=visible_xyz.device).type_as(visible_xyz) # world 2 cam
            K = torch.from_numpy(viewpoint_cam.K).to(device=visible_xyz.device).type_as(visible_xyz) 
            visible_xyz_cam = ((R @ visible_xyz) + T[None, :, None])
            pix = (K @ visible_xyz_cam).squeeze()
            pix /= pix[:, -1:]

            # -1 ~ 1        
            pix[:, 0] = (pix[:, 0]*2 - W)/W
            pix[:, 1] = (pix[:, 1]*2 - H)/H
            
            quat_init = torch.nn.functional.grid_sample(quat_from_norm.permute(1,0).reshape(1, -1, H, W), pix[None, None, :, :-1], mode='nearest', align_corners=True)
            quat_init = standardize_quaternion(quat_init) # torch.Size([1, 4, 1, n_vis])
            
            mask_zero_quat = (quat_init.abs().squeeze().sum(dim=-2)< 1e-9) # torch.Size([n_vis])
            quat_init_valid = quat_init[:, :, :,~mask_zero_quat].squeeze().permute(1,0) # -> quat_init_valid.shape: torch.Size([n_vis_val,4])
            # visibility_mark = visibility_mark.unsqueeze(dim=1)
            mask3d_visible_valid = visibility_mark.clone()
            mask3d_visible_valid[visibility_mark] *= ~mask_zero_quat # torch.Size([n_pnt])

            quaternion_new[mask3d_visible_valid] = torch.scatter(quaternion_new[mask3d_visible_valid], dim=1, index=idx_accum_quat[mask3d_visible_valid][:, None, None].repeat(1,1,4), src=quat_init_valid[:, None, :])
            
            idx_accum_quat[visibility_mark] +=1

            if idx_accum_quat.max().item() == max_memory:
                quaternion_new, idx_accum_quat = refresh_quaternion_new(quaternion_new, idx_accum_quat)

    quaternion_new = best_quaternion_new(quaternion_new, idx_accum_quat)
    quaternion_new = standardize_quaternion(quaternion_new)
    quaternion_new /= (torch.linalg.norm(quaternion_new, axis=-1, keepdims=True) + 1e-9)

    gaussians._rotation.data[: ,0] = copy.deepcopy(quaternion_new[:, 0].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,1] = copy.deepcopy(quaternion_new[:, 1].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,2] = copy.deepcopy(quaternion_new[:, 2].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,3] = copy.deepcopy(quaternion_new[:, 3].type_as(gaussians._rotation))
    
    gaussians._scaling[:, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))

    assert gaussians._rotation.data_ptr == ptr_gs_rot
    assert gaussians._scaling.data_ptr == ptr_gs_scale

    return gaussians

def training(dataset, opt, pipe, cfg_kitti, cfg_box, cfg_sd, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint_dir, debug_from, dyn_obj_list=['car'], exp_note='', run=None, args=None, output_dir=None):
    seed_all(dataset.seed)
    # add start / ending iteration of diffusion guidance for test
    testing_iterations.extend([cfg_sd.start_guiding_from_iter, cfg_sd.end_guiding_at_iter])
    
    first_iter = 0
    unique_str = prepare_output_and_logger(dataset, cfg_kitti, exp_note, output_dir=output_dir)
    if run is not None:
        run.tags += (unique_str, )
    # Set-up Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, cfg_kitti, cfg_box)
    gaussians.training_setup(opt)

    # emjay moved -----------------
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # --------------------------- 
    # emjay added ------------------
    if opt.normal_initialization:        
        gaussians = initialize_gaussians_with_window_normals(gaussians, scene, pipe, background)
        # gaussians = initialize_gaussians_with_normals(gaussians, scene, pipe, background)   
    # -----------------------------

    for instanceId in scene.gaussian_box_models.keys():
        scene.gaussian_box_models[instanceId].training_setup(opt)

    if checkpoint_dir:
        (model_params, first_iter) = torch.load(checkpoint_dir + "/chkpnt" + str(checkpoint_iterations[-1]) + ".pth")
        gaussians.restore(model_params, opt)
        for instanceId in scene.gaussian_box_models.keys():
            (model_params, first_iter) = torch.load(checkpoint_dir + "/chkpnt" + str(checkpoint_iterations[-1]) + f"_inst_{instanceId}" +".pth")
            scene.gaussian_box_models[instanceId].restore(model_params, opt)

    # Load diffusion regularizer
    if cfg_sd.reg_with_diffusion:
        from loss import LoRADiffusionRegularizer
        sd_reg = LoRADiffusionRegularizer(dataset, cfg_kitti, cfg_sd, opt.iterations)


    if cfg_sd.perceptual_loss:
        from loss import VGGPerceptualLoss
        perceptual_loss = VGGPerceptualLoss().cuda()


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
   

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        for instanceId in scene.gaussian_box_models.keys():
            scene.gaussian_box_models[instanceId].update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            for instanceId in scene.gaussian_box_models.keys():
                scene.gaussian_box_models[instanceId].oneupSHdegree()


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        all_bboxes = scene.getTrainBboxes() # This includes all existing vehicle bboxes in a scene that's "DYNAMIC".

        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # Retrieve frame, instance information on this camera 
        frame = viewpoint_cam.frame
        this_frame_includes_objects = False
        if frame in all_bboxes.keys():
            insts_in_frame = list(all_bboxes[frame].keys())
            this_frame_includes_objects = len(insts_in_frame) > 0 and (frame in all_bboxes.keys())
        
        # Retrieve GT image
        gt_image = viewpoint_cam.original_image
        
        # Render dynamic scene 
        if this_frame_includes_objects:
            bboxes = all_bboxes[frame]
            gaussians_boxes, box_models, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
            render_pkg = render_all(viewpoint_cam, gaussians, gaussians_boxes, box2worlds, pipe, background)
            # DEBUG
            # render_pkg_stat = render(viewpoint_cam, gaussians, pipe, background)
      
        # Render static scene
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        image, viewspace_point_tensor, visibility_filter, radii, cov_quat, cov_scale = render_pkg["render"], \
                                                                                       render_pkg["viewspace_points"], \
                                                                                       render_pkg["visibility_filter"], \
                                                                                       render_pkg["radii"], \
                                                                                       render_pkg['render_cov_quat'], \
                                                                                       render_pkg['render_cov_scale']         
        # DEBUG
        # if this_frame_includes_objects:
        #     torchvision.utils.save_image(render_pkg_stat["render"], "test_static.png")
        #     torchvision.utils.save_image(render_pkg["render"], "test_all.png")


        # Photometric loss  
        Ll1 = l1_loss(image, gt_image)
        lambda_dssim = opt.lambda_dssim
        # if iteration > cfg_sd.start_guiding_from_iter:
        #     lambda_dssim = opt.lambda_dssim_guidance
        # else:
        #     lambda_dssim = opt.lambda_dssim
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
        if opt.do_normal_guidance:
            Lng = loss_normal_guidance(viewpoint_cam, cov_quat, cov_scale)
            loss += opt.lambda_dnormal * Lng


        # Diffusion guidance loss
        if cfg_sd.reg_with_diffusion:
            if iteration > cfg_sd.start_guiding_from_iter and iteration < cfg_sd.end_guiding_at_iter:
                # [1] Augment viewpoints
                viewpoint_cam_aug, yaw, pitch, t_y, aug_dir = augmentCamera(viewpoint_cam, cfg_sd)
                
                # [2] Render augmented viewpoints (TODO: Should we regularize static scene only? for now, yes)
                image_aug = render(viewpoint_cam_aug, gaussians, pipe, background)["render"]

                # [3] Random crop renderings from augmented view.
                h_aug, w_aug = image_aug.shape[1], image_aug.shape[2]
                if cfg_sd.global_crop:
                    w_crop_start = randint(0, w_aug-h_aug)
                else:
                    if aug_dir == -1: # Look right
                        w_crop_start = randint((w_aug-h_aug)//2, w_aug-h_aug)
                    else: # Look left
                        w_crop_start = randint(0, (w_aug-h_aug) // 2)

                image_aug = image_aug[None, ..., w_crop_start:w_crop_start+h_aug]

                # [3] Compute guidance loss
                loss_guidance = sd_reg(image_aug, iteration)
                loss += loss_guidance

                # # [4] Compute optional VGG loss
                # if cfg_sd.perceptual_loss:
                #     loss_vgg = perceptual_loss(image_aug, gt_image[None, ..., w_crop_start:w_crop_start+h_aug])
                #     loss += cfg_sd.perceptual_loss_lambda * loss_vgg

        loss.backward()

        ## Do not update nan gradients for box optimizers 
        ## TODO: Why is this happening?
        if this_frame_includes_objects:
            for bm in box_models:
                if torch.any(torch.isnan(bm.delta_r.grad)) or torch.any(torch.isnan(bm.delta_s.grad)):
                    bm.delta_r.grad = torch.zeros_like(bm.delta_r.grad)
                    bm.delta_s.grad = torch.zeros_like(bm.delta_s.grad)
                    bm.delta_t.grad = torch.zeros_like(bm.delta_t.grad)
            
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss (L1)": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            scalar_kwargs={}
            scalar_kwargs["loss"] = loss.item()
            scalar_kwargs["loss_ema"] = ema_loss_for_log
            scalar_kwargs["l1_loss"] = Ll1.item()
            if opt.do_normal_guidance:
                scalar_kwargs["normal_loss"] = Lng.item()

            if cfg_sd.reg_with_diffusion and iteration > cfg_sd.start_guiding_from_iter and iteration < cfg_sd.end_guiding_at_iter:
                scalar_kwargs[f"{cfg_sd.guidance_mode}_loss_guidance"] = loss_guidance.item()

                # Record box refinment information 
                deltas = []
                if this_frame_includes_objects:
                    for box_model in box_models:
                        deltas.append(box_model.get_deltas())
                    deltas = torch.mean(torch.Tensor(deltas), dim=0)
                    scalar_kwargs["delta_r_norm"] = deltas[0].item()
                    scalar_kwargs["delta_s_norm"] = deltas[1].item()
                    scalar_kwargs["delta_t_norm"] = deltas[2].item()


        # Log and save
        with torch.no_grad():
            if not args.no_wandb:
                save_dir = None
                if dataset.save_results_as_images:
                    save_dir = scene.model_path
                wandb.log(scalar_kwargs, step=iteration)
                training_report(iteration, testing_iterations, scene, all_bboxes, gaussians, pipe, background, dyn_obj_list, cfg_sd=cfg_sd, scalar_kwargs=scalar_kwargs, save_dir=save_dir)
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        end_idx = gaussians.get_xyz.shape[0]
        cur_viewspace_point_tensor = slice_with_grad(viewspace_point_tensor, 0, end_idx)
        densification_and_optimization(gaussians, opt, cfg_sd, iteration, cur_viewspace_point_tensor, visibility_filter[:end_idx], scene, pipe, radii[:end_idx], dataset)
        
        if this_frame_includes_objects:
            start_idx=end_idx
            for gaussians_box in gaussians_boxes:
                # Optimize box gaussians
                idx_length = gaussians_box.get_xyz.shape[0]
                cur_viewspace_point_tensor = slice_with_grad(viewspace_point_tensor, start_idx, start_idx+idx_length)
                densification_and_optimization(gaussians_box, 
                                               opt,
                                               cfg_sd, 
                                               iteration, 
                                               cur_viewspace_point_tensor, 
                                               visibility_filter[start_idx:start_idx+idx_length], 
                                               scene, 
                                               pipe, 
                                               radii[start_idx:start_idx+idx_length],
                                               dataset,
                                               box=True)
                start_idx += idx_length

            # start_idx=0
            # for gaussians_box in gaussians_boxes:
            #     # Optimize box gaussians
            #     idx_length = gaussians_box.get_xyz.shape[0]
            #     cur_viewspace_point_tensor_dyn = slice_with_grad(viewspace_point_tensor_dyn, start_idx, start_idx+idx_length)
            #     densification_and_optimization(gaussians_box, 
            #                                    opt,
            #                                    cfg_sd, 
            #                                    iteration, 
            #                                    cur_viewspace_point_tensor_dyn, 
            #                                    visibility_filter_dyn[start_idx:start_idx+idx_length], 
            #                                    scene, 
            #                                    pipe, 
            #                                    radii_dyn[start_idx:start_idx+idx_length],
            #                                    dataset,
            #                                    box=True)
            #     start_idx += idx_length

            # Optimize bounding boxes
            for box_model in box_models:
                box_model.optimizer.step()
                box_model.optimizer.zero_grad()
                box_model.regularize(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                for instanceId in scene.gaussian_box_models.keys():
                    torch.save((scene.gaussian_box_models[instanceId].capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + f"_inst_{instanceId}" +".pth")


def slice_with_grad(tensor, start, end):
    out = tensor[start:end]
    out.grad = tensor.grad[start:end]
    return out

def densification_and_optimization(gaussians, opt, cfg_sd, iteration, viewspace_point_tensor, visibility_filter, scene, pipe, radii, dataset, box=False):
    # Densification
    if box:
        condition = iteration < opt.densify_until_iter_box
    else:
        condition = iteration < opt.densify_until_iter
    if condition: 
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            densify_grad_threshold = opt.densify_grad_threshold
            if box:
                densify_grad_threshold *= 0.5
                if size_threshold is not None:
                    size_threshold *= 0.5
            # do_prune = (iteration < cfg_sd.start_guiding_from_iter) and cfg_sd.do_prune
            # gaussians.densify_and_prune(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, prune=do_prune)
            gaussians.densify_and_prune(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        
        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()

    # Optimizer step
    if iteration < opt.iterations:
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)


def prepare_output_and_logger(args, cfg_kitti, exp_note, output_dir): 
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join(output_dir, f"{cfg_kitti.seq}_{str(cfg_kitti.start_frame).zfill(10)}_{str(cfg_kitti.end_frame).zfill(10)}", f"{unique_str[0:10]}_{exp_note}")
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    return unique_str

# emjay added -----------------------------------
def render_novelview_image(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, add_xrot_val=0, add_zrot_val=0, add_tz = 0):
    frame = viewpoint.frame
    this_frame_includes_objects = False
    if frame in all_bboxes.keys():
        insts_in_frame = list(all_bboxes[frame].keys())
        this_frame_includes_objects = len(insts_in_frame) > 0 and (frame in all_bboxes.keys())

    
    image_full = None
    viewpoint_new = make_camera_like_input_camera(viewpoint, add_xrot_val=add_xrot_val, add_zrot_val=add_zrot_val, add_tz = add_tz)    

    if this_frame_includes_objects:
        bboxes = all_bboxes[frame]
        gaussians_boxes, box_models, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
        image_full = torch.clamp(render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render"], 0.0, 1.0)
        image = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render"], 0.0, 1.0)
    else: 
        image = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render"], 0.0, 1.0)    

    # emjay added -------------------------
    # for debugging
    # from PIL import Image
    # import numpy as np
    # Image.fromarray((image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save('viewchange_yrot45_yt-10.png')
    # ---------------------------------------

    return image, image_full

def render_novelview_rotaxis(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, idx_best='min_scale', add_xrot_val=0, add_yrot_val=0, add_zrot_val=0, add_tx = 0, add_ty = 0, add_tz = 0):
    frame = viewpoint.frame
    normal_gt = viewpoint.original_normal.reshape(3,1,-1).permute(2,0,1)

    this_frame_includes_objects = False
    if frame in all_bboxes.keys():
        insts_in_frame = list(all_bboxes[frame].keys())
        this_frame_includes_objects = len(insts_in_frame) > 0 and (frame in all_bboxes.keys())

    cov_rot_full = None
    _, H, W = viewpoint.original_normal.shape    


    viewpoint_new = make_camera_like_input_camera(viewpoint, add_xrot_val=add_xrot_val, add_zrot_val=add_zrot_val, add_tz = add_tz)    

    if this_frame_includes_objects:
        bboxes = all_bboxes[frame]
        gaussians_boxes, box_models, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
        #cov_quat_full = torch.clamp(render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render_cov_quat"], -1.0, 1.0)
        cov_quat_full = render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render_cov_quat"]
        #cov_quat = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"], -1.0, 1.0)
        cov_quat = render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"]
        cov_scale_full = render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render_cov_scale"]
        cov_scale = render(viewpoint_new, gaussians, pipe, background)["render_cov_scale"]

        cov_quat_full = cov_quat_full.permute(1,2,0).reshape(-1, 4).contiguous()
        cov_rot_full = quaternion_to_matrix(cov_quat_full)                # cov_rot.shape: torch.Size([n_pix, 3, 3])
        cov_scale_full = cov_scale_full.permute(1,2,0).reshape(-1,3).contiguous()        
    else: 
        cov_quat = render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"]
        cov_scale = render(viewpoint_new, gaussians, pipe, background)["render_cov_scale"]
    
    cov_quat = cov_quat.permute(1,2,0).reshape(-1, 4).contiguous() # -> npix x 4
    cov_rot = quaternion_to_matrix(cov_quat)                # cov_rot.shape: torch.Size([n_pix, 3, 3])
    cov_scale = cov_scale.permute(1,2,0).reshape(-1,3).contiguous() # -> n_pix, 3

    # emjay added -------------------------
    # for debugging
    # from PIL import Image
    # import numpy as np
    # Image.fromarray((image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save('viewchange_yrot45_yt-10.png')
    # ---------------------------------------

    # cov_rot = cov_rot.permute(1,2,0)                # -> shape: torch.Size([3, 3, n_pix])
    # R_world2cam = torch.transpose(viewpoint.R, 0, 1)
    R_world2cam = torch.from_numpy(viewpoint.R).transpose(-1, -2).to(device=cov_rot.device).type_as(cov_rot)
    R_world2cam = R_world2cam[None] # -> 1x3x3
    norm_like = (R_world2cam @ cov_rot) # npix x 3 x 3    

    if idx_best == 'gt_like':
        idx_best = torch.argmax(torch.sum(normal_gt * norm_like, dim=1), dim=1)[:,None, None].repeat(1,3,1)    
    elif idx_best == 'min_scale':
        idx_best = torch.argmin(cov_scale, dim=1)[:,None, None].repeat(1,3,1)
    else:
        raise RuntimeError(f'unknown idx_best:{idx_best}')

    norm_like_best = norm_like.gather(dim=2, index = idx_best).squeeze().permute(1,0)
    norm_like_best = ((( norm_like_best.reshape(-1, H, W)*-1)+1)/2)*255
    norm_like_best = torch.clip(norm_like_best, min=0, max=255)
    norm_like_best = norm_like_best.to(torch.uint8)  

    norm_like_best_full = None
    if this_frame_includes_objects:
        norm_like_full = (R_world2cam @ cov_rot) # npix x 3 x 3    
        idx_best = torch.argmax(torch.sum(normal_gt * norm_like_full, dim=1), dim=1)[:,None, None].repeat(1,3,1)
        norm_like_best_full = norm_like.gather(dim=2, index = idx_best).squeeze().permute(1,0)
        norm_like_best_full = ((( norm_like_best_full.reshape(-1, H, W)*-1)+1)/2)*255
        norm_like_best_full = torch.clip(norm_like_best_full, min=0, max=255)
        norm_like_best_full = norm_like_best_full.to(torch.uint8)   

    return norm_like_best, norm_like_best_full

def render_novelview_bestrotaxis(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, add_xrot_val=0, add_yrot_val=0, add_zrot_val=0, add_tx = 0, add_ty = 0, add_tz = 0):
    frame = viewpoint.frame
    normal_gt = viewpoint.original_normal.reshape(3,1,-1).permute(2,0,1)

    this_frame_includes_objects = False
    if frame in all_bboxes.keys():
        insts_in_frame = list(all_bboxes[frame].keys())
        this_frame_includes_objects = len(insts_in_frame) > 0 and (frame in all_bboxes.keys())

    cov_rot_full = None
    _, H, W = viewpoint.original_normal.shape    

   
    viewpoint_new = make_camera_like_input_camera(viewpoint, add_xrot_val=add_xrot_val, add_zrot_val=add_zrot_val, add_tz = add_tz)    
    
    if this_frame_includes_objects:
        bboxes = all_bboxes[frame]
        gaussians_boxes, boxmodels, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
        cov_quat_full = torch.clamp(render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render_cov_quat"], -1.0, 1.0)
        cov_quat = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"], -1.0, 1.0)

        cov_quat_full = cov_quat_full.permute(1,2,0).reshape(-1, 4).contiguous()
        cov_rot_full = quaternion_to_matrix(cov_quat_full)                # cov_rot.shape: torch.Size([n_pix, 3, 3])
    else: 
        cov_quat = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"], -1.0, 1.0)    
    
    cov_quat = cov_quat.permute(1,2,0).reshape(-1, 4).contiguous() # -> npix x 4
    cov_rot = quaternion_to_matrix(cov_quat)                # cov_rot.shape: torch.Size([n_pix, 3, 3])

    # emjay added -------------------------
    # for debugging
    # from PIL import Image
    # import numpy as np
    # Image.fromarray((image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save('viewchange_yrot45_yt-10.png')
    # ---------------------------------------

    # cov_rot = cov_rot.permute(1,2,0)                # -> shape: torch.Size([3, 3, n_pix])
    # R_world2cam = torch.transpose(viewpoint.R, 0, 1)
    R_world2cam = torch.from_numpy(viewpoint.R).transpose(-1, -2).to(device=cov_rot.device).type_as(cov_rot)
    R_world2cam = R_world2cam[None] # -> 1x3x3


    norm_like = (R_world2cam @ cov_rot) # npix x 3 x 3    
    idx_best = torch.argmax(torch.sum(normal_gt * norm_like, dim=1), dim=1)[:,None, None].repeat(1,3,1)
    norm_like_best = norm_like.gather(dim=2, index = idx_best).squeeze().permute(1,0)
    norm_like_best = ((( norm_like_best.reshape(-1, H, W)*-1)+1)/2)*255
    norm_like_best = torch.clip(norm_like_best, min=0, max=255)
    norm_like_best = norm_like_best.to(torch.uint8)  

    norm_like_best_full = None
    if this_frame_includes_objects:
        norm_like_full = (R_world2cam @ cov_rot) # npix x 3 x 3    
        idx_best = torch.argmax(torch.sum(normal_gt * norm_like_full, dim=1), dim=1)[:,None, None].repeat(1,3,1)
        norm_like_best_full = norm_like.gather(dim=2, index = idx_best).squeeze().permute(1,0)
        norm_like_best_full = ((( norm_like_best_full.reshape(-1, H, W)*-1)+1)/2)*255
        norm_like_best_full = torch.clip(norm_like_best_full, min=0, max=255)
        norm_like_best_full = norm_like_best_full.to(torch.uint8)   

    return norm_like_best, norm_like_best_full

def render_novelview_rotaxis_onebyone(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, add_xrot_val=0, add_yrot_val=0, add_zrot_val=0, add_tx = 0, add_ty = 0, add_tz = 0):
    frame = viewpoint.frame
    this_frame_includes_objects = False
    if frame in all_bboxes.keys():
        insts_in_frame = list(all_bboxes[frame].keys())
        this_frame_includes_objects = len(insts_in_frame) > 0 and (frame in all_bboxes.keys())

    cov_rot_full = None
    _, H, W = viewpoint.original_normal.shape    


    viewpoint_new = make_camera_like_input_camera(viewpoint, add_xrot_val=add_xrot_val, add_zrot_val=add_zrot_val, add_tz=add_tz)    

    if this_frame_includes_objects:
        bboxes = all_bboxes[frame]
        gaussians_boxes, boxmodels, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
        cov_quat_full = torch.clamp(render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render_cov_quat"], -1.0, 1.0)
        cov_quat = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"], -1.0, 1.0)

        cov_quat_full = cov_quat_full.permute(1,2,0).reshape(-1, 4).contiguous()
        cov_rot_full = quaternion_to_matrix(cov_quat_full)                # cov_rot.shape: torch.Size([n_pix, 3, 3])
    else: 
        cov_quat = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"], -1.0, 1.0)    
    
    cov_quat = cov_quat.permute(1,2,0).reshape(-1, 4).contiguous()
    cov_rot = quaternion_to_matrix(cov_quat)                # cov_rot.shape: torch.Size([n_pix, 3, 3])

    # emjay added -------------------------
    # for debugging
    # from PIL import Image
    # import numpy as np
    # Image.fromarray((image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save('viewchange_yrot45_yt-10.png')
    # ---------------------------------------

    cov_rot = cov_rot.permute(1,2,0)                # -> shape: torch.Size([3, 3, n_pix])
    # R_world2cam = torch.transpose(viewpoint.R, 0, 1)
    R_world2cam = torch.from_numpy(viewpoint.R).transpose(-1, -2).to(device=cov_rot.device).type_as(cov_rot)

    cov_axis_list = []
    for i in range(3):
        norm_like_cam = (R_world2cam @ cov_rot[:, i, :])
        cov_axis = ((( norm_like_cam.reshape(-1, H, W)*-1)+1)/2)*255
        cov_axis = torch.clip(cov_axis, min=0, max=255)
        cov_axis = cov_axis.to(torch.uint8)  
        cov_axis_list.append(cov_axis) 

    cov_axis_y = (( (R_world2cam @ cov_rot[:, 1, :]).reshape(-1, H, W)*-1)+1)/2
    cov_axis_z = (( (R_world2cam @ cov_rot[:, 2, :]).reshape(-1, H, W)*-1)+1)/2

    cov_axis_full_list = [None, None, None]    
    if this_frame_includes_objects:
        cov_rot_full  = cov_rot_full.permute(1,2,0)  # -> shape: torch.Size([3, 3, n_pix])
        for i in range(3):
            norm_like_cam = (R_world2cam @ cov_rot_full[:, i, :])
            cov_axis_full = ((( norm_like_cam.reshape(-1, H, W)*-1)+1)/2) * 255
            cov_axis_full = torch.clip(cov_axis_full, min=0, max=255)
            cov_axis_full = cov_axis_full.to(torch.uint8)
            cov_axis_full_list.append(cov_axis_full) 

    return cov_axis_list, cov_axis_full_list



# def training_report(iteration, testing_iterations, scene : Scene, all_bboxes, gaussians, pipe, background, dyn_obj_list, cfg_sd=None, viewpoint_stack=None, scalar_kwargs=None, save_dir=None):
#     # Report test and samples of training set
#     if iteration in testing_iterations:
#         torch.cuda.empty_cache()
#         validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
#                               {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
#         if viewpoint_stack is not None:
#             validation_configs = ({'name': 'test', 'cameras' : viewpoint_stack}, 
#                                   {'name': 'train', 'cameras' : viewpoint_stack})

#         # [-30, 0,   0,   0,   0.5,   0], \
#         # [-15, 0,   0,   0,   0.5,   0], \
#         # [-7,  0,   0,   0,   0.5,   0], \
#         # add augmented image to log candidates - down
#                           #Rx  Rz  Tz
#         cam_aug_params = [[0,  30,  0], \
#                           [0, -30,  0], \
#                           [0, 60,  0], \
#                           [0, -60,  0]]
#         # view down & move up 
#         cam_aug_params += [[-i, 0, i/15*1.5] for i in range(17)]


#         for config in validation_configs:
#             if config['cameras'] and len(config['cameras']) > 0:
#                 l1_test = 0.0
#                 psnr_test = 0.0
                
#                 if save_dir is not None and iteration == testing_iterations[-1]:
#                 # if save_dir is not None and iteration in testing_iterations:
#                     viz_types = ['render_rgb', 'render_axis_min_scale', 'render_axis_gt_like', 'render_rgb_aug']
#                     viz_dirs = {}
#                     for viz_type in viz_types:
#                         viz_dir = os.path.join(save_dir, 'results', config['name'], viz_type, str(iteration))
#                         viz_dirs[viz_type] = viz_dir
#                         os.makedirs(viz_dir, exist_ok=True)

#                     # for cam_aug_param in cam_aug_params:
#                     #     viz_dir_key = f"Rx{cam_aug_param[0]}_Ry{cam_aug_param[1]}_Rz{cam_aug_param[2]}_tx{cam_aug_param[3]}_ty{cam_aug_param[4]}_tz{cam_aug_param[5]}"
#                     #     viz_dir = os.path.join(save_dir, 'results', config['name'], 'render_rgb_aug', str(iteration), viz_dir_key)
#                     #     viz_dirs['render_rgb_aug'][viz_dir_key] = viz_dir
#                     #     os.makedirs(viz_dir, exist_ok=True)


#                 pbar = tqdm(config['cameras'], total=len(config['cameras']))
#                 pbar.set_description(f"Rendering {config['name']} images")
#                 for idx, viewpoint in tqdm(enumerate(config['cameras']), total=len(config['cameras'])):
#                     wandb_cond = (idx % 10 ==0)
#                     log_imgs = [] 

#                     # add rendered image to log candidates
#                     image, image_full = render_novelview_image(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene)
#                     gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    
#                     # do evaluation
#                     l1_test += l1_loss(image, gt_image).mean().double()
#                     psnr_test += psnr(image, gt_image).mean().double()
                    
#                     # log images
#                     if save_dir is not None and iteration == testing_iterations[-1]:
#                     # if save_dir is not None and iteration in testing_iterations:
#                         torchvision.utils.save_image(image, os.path.join(viz_dirs['render_rgb'], viewpoint.image_name))
#                         if image_full is not None:
#                             torchvision.utils.save_image(image_full, os.path.join(viz_dirs['render_rgb'], f"{viewpoint.image_name[:-4]}_with_objects.png"))
#                     # if wandb_cond:
#                     #     log_imgs.append(wandb.Image(image[None], caption="render"))
#                     #     if image_full is not None:
#                     #         log_imgs.append(wandb.Image(image_full[None], caption="render_with_objects"))   

#                     # add normal image to log candidates
#                     gt_norm_rgb = ((viewpoint.original_normal*-1 + 1) * 0.5) * 255 
#                     gt_norm_rgb = torch.clip(gt_norm_rgb, min=0, max=255)
#                     gt_norm_rgb = gt_norm_rgb.to(torch.uint8)                    
#                     # if wandb_cond:
#                     #     log_imgs.append(wandb.Image(gt_norm_rgb[None], caption="normal"))

#                     # add cov rot axis to log candidates                                                
#                     if save_dir is not None and iteration == testing_iterations[-1]:
#                     # if save_dir is not None and iteration in testing_iterations:
#                         render_axis_best, render_axis_best_full = render_novelview_rotaxis(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, idx_best = 'min_scale')
#                         torchvision.utils.save_image(render_axis_best / 255, os.path.join(viz_dirs['render_axis_min_scale'], viewpoint.image_name))
#                         if render_axis_best_full is not None:
#                             torchvision.utils.save_image(render_axis_best_full / 255, os.path.join(viz_dirs['render_axis_min_scale'], f"{viewpoint.image_name[:-4]}_with_objects.png"))
#                     # if wandb_cond:
#                     #     log_imgs.append(wandb.Image(render_axis_best[None], caption="render_axis_min_scale"))
#                     #     if render_axis_best_full is not None:
#                     #         log_imgs.append(wandb.Image(render_axis_best_full[None], caption="render_axis_min_scale"))

#                     if save_dir is not None and iteration == testing_iterations[-1]:
#                     # if save_dir is not None and iteration in testing_iterations:
#                         render_axis_best, render_axis_best_full = render_novelview_rotaxis(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, idx_best = 'gt_like')
#                         torchvision.utils.save_image(render_axis_best / 255, os.path.join(viz_dirs['render_axis_gt_like'], viewpoint.image_name))
#                         if render_axis_best_full is not None:
#                             torchvision.utils.save_image(render_axis_best_full / 255, os.path.join(viz_dirs['render_axis_gt_like'], f"{viewpoint.image_name[:-4]}_with_objects.png"))
#                     # if wandb_cond:
#                     #     log_imgs.append(wandb.Image(render_axis_best[None], caption="render_axis_gt_like"))
#                     #     if render_axis_best_full is not None:
#                     #         log_imgs.append(wandb.Image(render_axis_best_full[None], caption="render_axis_gt_like"))

#                     # render_rot, render_rot_full = render_novelview_rotaxis(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene)                    
#                     # log_imgs.append(wandb.Image(render_rot[0][None], caption="render_rot_x"))
#                     # log_imgs.append(wandb.Image(render_rot[1][None], caption="render_rot_y"))
#                     # log_imgs.append(wandb.Image(render_rot[2][None], caption="render_rot_z"))
#                     # if render_rot_full[0] is not None:
#                     #     log_imgs.append(wandb.Image(render_rot_full[0][None], caption="render_rot_full_x"))
#                     #     log_imgs.append(wandb.Image(render_rot_full[1][None], caption="render_rot_full_y"))
#                     #     log_imgs.append(wandb.Image(render_rot_full[2][None], caption="render_rot_full_z"))

#                     # add gt image to log candidates
#                     # if wandb_cond:
#                     #     if iteration == testing_iterations[-1]:
#                     #         log_imgs.append(wandb.Image(gt_image[None], caption="ground_truth"))
                    

#                     if save_dir is not None and iteration == testing_iterations[-1]:
#                     # if save_dir is not None and iteration in testing_iterations:
#                         for rx, rz, tz in cam_aug_params:
#                             aug_caption = f"Rx: {rx}| Rz: {rz} | tz: {tz}"
#                             image, image_full = render_novelview_image(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, rx, rz, tz)
#                             torchvision.utils.save_image(image, os.path.join(viz_dirs['render_rgb_aug'], viewpoint.image_name[:-4] + f"_Rx{rx}_Rz{rz}_tz{tz}.png"))
#                                 # if image_full is not None:
#                                 #     torchvision.utils.save_image(image_full, os.path.join(viz_dirs['render_rgb_aug'], viewpoint.image_name[:-4] + f"_Rx{rx}_Ry{ry}_Rz{rz}_tx{tx}_ty{ty}_tz{tz}_with_objects.png"))
#                             # if wandb_cond:
#                             #     log_imgs.append(wandb.Image(image[None], caption="render - " + aug_caption))
#                             #     if image_full is not None:
#                             #         log_imgs.append(wandb.Image(image_full[None], caption="render_with_objects"+aug_caption))
                    

#                     # log everything
#                     if wandb_cond:
#                         wandb.log({config['name'] + f"_view_{viewpoint.image_name}":log_imgs}, step=iteration)   

                    

#                 psnr_test /= len(config['cameras'])
#                 l1_test /= len(config['cameras'])          
#                 print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
#                 # emjay modified -----------------------
#                 wandb.log({ config['name'] + '/loss_viewpoint - l1_loss': l1_test, 
#                             config['name'] + '/loss_viewpoint - psnr': psnr_test}, 
#                             step=iteration)                
#                 # original ----------------------------
#                 # if tb_writer:
#                 #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
#                 #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
#                 # -------------------------------------

#         # emjay modified -------------------------------------
#         wandb.log({"scene/opacity_histogram": wandb.Histogram(scene.gaussians.get_opacity.squeeze().tolist())}, step=iteration)        
#         wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]}, step = iteration)
#         # original --------------------------------------------
#         # if tb_writer:
#         #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
#         #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
#         # ------------------------------------------------------
#         torch.cuda.empty_cache()

# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Training script parameters")
#     lp = ModelParams(parser)
#     op = OptimizationParams(parser)
#     pp = PipelineParams(parser)
#     dp = KITTI360DataParams(parser)
#     bp = BoxModelParams(parser)
#     sp = SDRegularizationParams(parser) 

#     parser.add_argument('--ip', type=str, default="127.0.0.1")
#     parser.add_argument('--exp_note', type=str, default="")
#     parser.add_argument('--port', type=int, default=6009)
#     parser.add_argument('--debug_from', type=int, default=-1)
#     parser.add_argument('--detect_anomaly', action='store_true', default=False)
#     parser.add_argument('--no_wandb', action='store_true', default=False)
#     # parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000, 70_000, 100_000])
#     parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
#     # parser.add_argument("--save_iterations", nargs="+", type=int, default=[100_000])
#     parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
#     parser.add_argument("--quiet", action="store_true")
#     parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
#     parser.add_argument("--start_checkpoint_dir", type=str, default = None)
#     args = parser.parse_args(sys.argv[1:])
#     args.save_iterations.append(args.iterations)

#     # initialize wandb -------------------
#     if not args.no_wandb:
#         dp_cache = dp.extract(args)
#         lp_cache = lp.extract(args)
#         cur_data = "_" + dp_cache.seq + "_" + str(int(dp_cache.start_frame)) + "_" + str(int(dp_cache.end_frame))
#         run = wandb.init(        
#             project="3dgs_kitti",       # Set the project where this run will be logged
#             name = args.exp_note,       # exp name
#             tags = args.exp_note.split('_') + [dp_cache.seq] + [str(dp_cache.start_frame)] + [str(dp_cache.end_frame)] + [lp_cache.data_type], 
#             config = parser             # Track hyperparameters and run metadata        
#         )
#         assert run is wandb.run    
#     else:
#         run = None
#     # -----------------------------------


#     print("Optimizing " + args.model_path)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     # Start GUI server, configure and run training
#     # network_gui.init(args.ip, args.port)
#     torch.autograd.set_detect_anomaly(args.detect_anomaly)
#     training(lp.extract(args), 
#              op.extract(args), 
#              pp.extract(args), 
#              dp.extract(args),
#              bp.extract(args),
#              sp.extract(args), 
#              args.test_iterations, 
#              args.save_iterations, 
#              args.checkpoint_iterations, 
#              args.start_checkpoint_dir, 
#              args.debug_from, 
#              exp_note=args.exp_note,
#              run=run,
#              args=args,
#              output_dir=args.output_dir)

#     # All done
#     print("\nTraining complete.")