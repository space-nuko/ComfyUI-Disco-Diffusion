import pathlib, shutil, os, sys
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
import json
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import random
import hashlib
from functools import partial
from numpy import asarray
import time
import warnings
import os

import comfy.model_management
from . import disco_utils
from .settings import DiscoDiffusionSettings
from .do_run import do_run


# %%
# !! {"metadata":{
# !!   "id": "DoTheRun"
# !! }}
#@title Do the Run!
#@markdown `n_batches` ignored with animation modes.
def diffuse(clip, args: DiscoDiffusionSettings, batchNum):
    args.display_rate = 20 #@param{type: 'number'}
    args.n_batches = 50 #@param{type: 'number'}

    if args.animation_mode == 'Video Input':
        args.steps = args.video_init_steps

    #Update Model Settings
    timestep_respacing = f'ddim{args.steps}'
    diffusion_steps = (1000//args.steps)*args.steps if args.steps < 1000 else args.steps
    args.MS.model_config.update({
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
    })

    args.batch_size = 1

    def move_files(start_num, end_num, old_folder, new_folder):
        for i in range(start_num, end_num):
            old_file = old_folder + f'/{args.batch_name}({batchNum})_{i:04}.png'
            new_file = new_folder + f'/{args.batch_name}({batchNum})_{i:04}.png'
            os.rename(old_file, new_file)

    #@markdown ---


    args.resume_run = False #@param{type: 'boolean'}
    run_to_resume = 'latest' #@param{type: 'string'}
    resume_from_frame = 'latest' #@param{type: 'string'}
    retain_overwritten_frames = False #@param{type: 'boolean'}
    if retain_overwritten_frames:
        retainFolder = f'{args.batchFolder}/retained'
        os.makedirs(retainFolder, exist_ok=True)

    skip_step_ratio = int(args.frames_skip_steps.rstrip("%")) / 100
    args.calc_frames_skip_steps = math.floor(args.steps * skip_step_ratio)

    if args.animation_mode == 'Video Input':
        frames = sorted(glob(args.in_path+'/*.*'));
        if len(frames)==0:
            sys.exit("ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell.")
        flows = glob(args.flo_folder+'/*.*')
        if (len(flows)==0) and args.video_init_flow_warp:
            sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")

    if args.steps <= args.calc_frames_skip_steps:
        sys.exit("ERROR: You can't skip more steps than your total steps")

    if args.resume_run:
        if run_to_resume == 'latest':
            try:
                batchNum
            except:
                batchNum = len(glob(f"{args.batchFolder}/{args.batch_name}(*)_settings.txt"))-1
        else:
            batchNum = int(run_to_resume)
        if resume_from_frame == 'latest':
            start_frame = len(glob(args.batchFolder+f"/{args.batch_name}({batchNum})_*.png"))
            if args.animation_mode != '3D' and args.turbo_mode == True and start_frame > args.turbo_preroll and start_frame % int(args.turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(args.turbo_steps))
        else:
            start_frame = int(resume_from_frame)+1
            if args.animation_mode != '3D' and args.turbo_mode == True and start_frame > args.turbo_preroll and start_frame % int(args.turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(args.turbo_steps))
            if retain_overwritten_frames is True:
                existing_frames = len(glob(args.batchFolder+f"/{args.batch_name}({batchNum})_*.png"))
                frames_to_save = existing_frames - start_frame
                print(f'Moving {frames_to_save} frames to the Retained folder')
                move_files(start_frame, existing_frames, args.batchFolder, retainFolder)
    else:
        start_frame = 0
        batchNum = len(glob(args.batchFolder+"/*.txt"))
        while os.path.isfile(f"{args.batchFolder}/{args.batch_name}({batchNum})_settings.txt") or os.path.isfile(f"{args.batchFolder}/{args.batch_name}-{batchNum}_settings.txt"):
            batchNum += 1

    print(f'Starting Run: {args.batch_name}({batchNum}) at frame {start_frame}')

    if args.set_seed == 'random_seed':
        random.seed()
        seed = random.randint(0, 2**32)
        # print(f'Using seed: {seed}')
    else:
        seed = int(args.set_seed)

    args.n_batches = args.n_batches if args.animation_mode == 'None' else 1
    args.max_frames = args.max_frames if args.animation_mode == 'None' else 1
    args.start_frame = start_frame
    args.seed = seed
    args.prompts_series = disco_utils.split_prompts(args.text_prompts, args.max_frames) if args.text_prompts else None,
    args.image_prompts_series = disco_utils.split_prompts(args.image_prompts, args.max_frames) if args.image_prompts else None,

    # args = {
    #     'batchNum': batchNum,
    #     'prompts_series':split_prompts(text_prompts) if text_prompts else None,
    #     'image_prompts_series':split_prompts(image_prompts) if image_prompts else None,
    #     'seed': seed,
    #     'display_rate':display_rate,
    #     'n_batches':n_batches if animation_mode == 'None' else 1,
    #     'batch_size':batch_size,
    #     'batch_name': batch_name,
    #     'steps': steps,
    #     'diffusion_sampling_mode': diffusion_sampling_mode,
    #     'width_height': width_height,
    #     'clip_guidance_scale': clip_guidance_scale,
    #     'tv_scale': tv_scale,
    #     'range_scale': range_scale,
    #     'sat_scale': sat_scale,
    #     'cutn_batches': cutn_batches,
    #     'init_image': init_image,
    #     'init_scale': init_scale,
    #     'skip_steps': skip_steps,
    #     'side_x': side_x,
    #     'side_y': side_y,
    #     'timestep_respacing': timestep_respacing,
    #     'diffusion_steps': diffusion_steps,
    #     'animation_mode': animation_mode,
    #     'video_init_path': video_init_path,
    #     'extract_nth_frame': extract_nth_frame,
    #     'video_init_seed_continuity': video_init_seed_continuity,
    #     'key_frames': key_frames,
    #     'max_frames': max_frames if animation_mode != "None" else 1,
    #     'interp_spline': interp_spline,
    #     'start_frame': start_frame,
    #     'angle': angle,
    #     'zoom': zoom,
    #     'translation_x': translation_x,
    #     'translation_y': translation_y,
    #     'translation_z': translation_z,
    #     'rotation_3d_x': rotation_3d_x,
    #     'rotation_3d_y': rotation_3d_y,
    #     'rotation_3d_z': rotation_3d_z,
    #     'midas_depth_model': midas_depth_model,
    #     'midas_weight': midas_weight,
    #     'near_plane': near_plane,
    #     'far_plane': far_plane,
    #     'fov': fov,
    #     'padding_mode': padding_mode,
    #     'sampling_mode': sampling_mode,
    #     'angle_series':angle_series,
    #     'zoom_series':zoom_series,
    #     'translation_x_series':translation_x_series,
    #     'translation_y_series':translation_y_series,
    #     'translation_z_series':translation_z_series,
    #     'rotation_3d_x_series':rotation_3d_x_series,
    #     'rotation_3d_y_series':rotation_3d_y_series,
    #     'rotation_3d_z_series':rotation_3d_z_series,
    #     'frames_scale': frames_scale,
    #     'skip_step_ratio': skip_step_ratio,
    #     'calc_frames_skip_steps': calc_frames_skip_steps,
    #     'text_prompts': text_prompts,
    #     'image_prompts': image_prompts,
    #     'cut_overview': eval(cut_overview),
    #     'cut_innercut': eval(cut_innercut),
    #     'cut_ic_pow': eval(cut_ic_pow),
    #     'cut_icgray_p': eval(cut_icgray_p),
    #     'intermediate_saves': intermediate_saves,
    #     'intermediates_in_subfolder': intermediates_in_subfolder,
    #     'steps_per_checkpoint': steps_per_checkpoint,
    #     'perlin_init': perlin_init,
    #     'perlin_mode': perlin_mode,
    #     'set_seed': set_seed,
    #     'eta': eta,
    #     'clamp_grad': clamp_grad,
    #     'clamp_max': clamp_max,
    #     'skip_augs': skip_augs,
    #     'randomize_class': randomize_class,
    #     'clip_denoised': clip_denoised,
    #     'fuzzy_prompt': fuzzy_prompt,
    #     'rand_mag': rand_mag,
    #     'turbo_mode':turbo_mode,
    #     'turbo_steps':turbo_steps,
    #     'turbo_preroll':turbo_preroll,
    #     'use_vertical_symmetry': use_vertical_symmetry,
    #     'use_horizontal_symmetry': use_horizontal_symmetry,
    #     'transformation_percent': transformation_percent,
    #     #video init settings
    #     'video_init_steps': video_init_steps,
    #     'video_init_clip_guidance_scale': video_init_clip_guidance_scale,
    #     'video_init_tv_scale': video_init_tv_scale,
    #     'video_init_range_scale': video_init_range_scale,
    #     'video_init_sat_scale': video_init_sat_scale,
    #     'video_init_cutn_batches': video_init_cutn_batches,
    #     'video_init_skip_steps': video_init_skip_steps,
    #     'video_init_frames_scale': video_init_frames_scale,
    #     'video_init_frames_skip_steps': video_init_frames_skip_steps,
    #     #warp settings
    #     'video_init_flow_warp':video_init_flow_warp,
    #     'video_init_flow_blend':video_init_flow_blend,
    #     'video_init_check_consistency':video_init_check_consistency,
    #     'video_init_blend_mode':video_init_blend_mode
    # }

    # if animation_mode == 'Video Input':
    #     # This isn't great in terms of what will get saved to the settings.. but it should work.
    #     args['steps'] = args['video_init_steps']
    #     args['clip_guidance_scale'] = args['video_init_clip_guidance_scale']
    #     args['tv_scale'] = args['video_init_tv_scale']
    #     args['range_scale'] = args['video_init_range_scale']
    #     args['sat_scale'] = args['video_init_sat_scale']
    #     args['cutn_batches'] = args['video_init_cutn_batches']
    #     args['skip_steps'] = args['video_init_skip_steps']
    #     args['frames_scale'] = args['video_init_frames_scale']
    #     args['frames_skip_steps'] = args['video_init_frames_skip_steps']

    # args = SimpleNamespace(**args)

    device = comfy.model_management.get_torch_device()

    print('Prepping model...')
    model, diffusion = create_model_and_diffusion(**args.MS.model_config)
    if args.MS.diffusion_model == 'custom':
        model.load_state_dict(torch.load(args.MS.custom_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(f'{args.MS.model_path}/{args.MS.get_model_filename(args.MS.diffusion_model)}', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if args.MS.model_config['use_fp16']:
        model.convert_to_fp16()

    gc.collect()
    torch.cuda.empty_cache()
    try:
        do_run(diffusion, model, clip, args, batchNum)
    except KeyboardInterrupt:
        pass
    finally:
        print('Seed used:', seed)
        gc.collect()
        torch.cuda.empty_cache()
