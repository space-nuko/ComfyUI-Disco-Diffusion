import shutil
import cv2
import pandas as pd
import gc
import math
import lpips
import PIL
from PIL import Image, ImageOps
import requests
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
from numpy import asarray
from . import py3d_tools as p3dT
from . import disco_xform_utils as dxf
from .CLIP import clip
import open_clip

import comfy.model_management
import comfy.utils
from comfy.clip_vision import ClipVisionModel
import comfy.sd

from . import disco_utils
from .make_cutouts import MakeCutouts, MakeCutoutsDango
from .midas_model import init_midas_depth_model
from .settings import DiscoDiffusionSettings

# Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
stop_on_next_loop = False
TRANSLATION_SCALE = 1.0/200.0

def encode_text(clip_model, prompt):
    if isinstance(clip_model, comfy.sd.CLIP):
        # ComfyUI
        _, cond_pooled = clip_model.encode_from_tokens(clip_model.tokenize(prompt), return_pooled=True)
        return cond_pooled.float()
    else:
        # OpenAI/OpenClip
        device = comfy.model_management.get_torch_device()
        return clip_model.encode_text(clip.tokenize([prompt]).to(device)).float()

def encode_images(clip_vision, images):
    if isinstance(clip_vision, ClipVisionModel):
        # ComfyUI
        return clip_vision.model(pixel_values=images).image_embeds.float()
    else:
        # OpenAI/OpenClip
        return clip_vision.encode_image(images).float()

def do_3d_step(args: DiscoDiffusionSettings, img_filepath, frame_num, midas_model, midas_transform):
    if args.key_frames:
        translation_x = args.translation_x_series[frame_num]
        translation_y = args.translation_y_series[frame_num]
        translation_z = args.translation_z_series[frame_num]
        rotation_3d_x = args.rotation_3d_x_series[frame_num]
        rotation_3d_y = args.rotation_3d_y_series[frame_num]
        rotation_3d_z = args.rotation_3d_z_series[frame_num]
        print(
            f'translation_x: {translation_x}',
            f'translation_y: {translation_y}',
            f'translation_z: {translation_z}',
            f'rotation_3d_x: {rotation_3d_x}',
            f'rotation_3d_y: {rotation_3d_y}',
            f'rotation_3d_z: {rotation_3d_z}',
        )

    device = comfy.model_management.get_torch_device()

    translate_xyz = [-translation_x*TRANSLATION_SCALE, translation_y *
                     TRANSLATION_SCALE, -translation_z*TRANSLATION_SCALE]
    rotate_xyz_degrees = [rotation_3d_x, rotation_3d_y, rotation_3d_z]
    print('translation:', translate_xyz)
    print('rotation:', rotate_xyz_degrees)
    rotate_xyz = [math.radians(rotate_xyz_degrees[0]), math.radians(
        rotate_xyz_degrees[1]), math.radians(rotate_xyz_degrees[2])]
    rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(
        rotate_xyz, device=device), "XYZ").unsqueeze(0)
    print("rot_mat: " + str(rot_mat))
    next_step_pil = dxf.transform_image_3d(img_filepath, midas_model, midas_transform, device,
                                           rot_mat, translate_xyz, args.near_plane, args.far_plane,
                                           args.fov, padding_mode=args.padding_mode,
                                           sampling_mode=args.sampling_mode, midas_weight=args.midas_weight)
    return next_step_pil


def horiz_symmetry(x):
    [n, c, h, w] = x.size()
    x = torch.concat(
        (x[:, :, :, :w//2], torch.flip(x[:, :, :, :w//2], [-1])), -1)
    print("horizontal symmetry applied")
    return x


def vert_symmetry(x):
    [n, c, h, w] = x.size()
    x = torch.concat(
        (x[:, :, :h//2, :], torch.flip(x[:, :, :h//2, :], [-2])), -2)
    print("vertical symmetry applied")
    return x


def id(x):
    return x


def get_input_resolution(clip_model):
    # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
    try:
        if isinstance(clip_model, ClipVisionModel):
            # ComfyUI (Transformers)
            return clip_model.model.config.image_size
        elif isinstance(clip_model, open_clip.CLIP):
            # OpenClip
            return clip_model.visual.image_size[0]
        else:
            # OpenAI
            return clip_model.visual.input_resolution
    except Exception as err:
        print("Couldn't find clip vision image size! " + str(err) + " " + str(type(clip_model)))
        return 224


def do_run(diffusion, model, clip_model, clip_vision, args: DiscoDiffusionSettings, batchNum):
    global stop_on_next_loop

    print(range(args.start_frame, args.max_frames))

    pbar = comfy.utils.ProgressBar(diffusion.num_timesteps - args.skip_steps)

    midas_model = None
    midas_transform = None
    midas_net_w = None
    midas_net_h = None
    midas_resize_mode = None
    midas_normalization = None

    if (args.animation_mode == "3D") and (args.midas_weight > 0.0):
        midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(
            args.midas_depth_model)

    results = []

    for frame_num in range(args.start_frame, args.max_frames):
        if stop_on_next_loop:
            break
        with torch.inference_mode(False):
            results += run_one_frame(diffusion, model, clip_model, clip_vision, args, batchNum, frame_num, pbar,
                                     midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization)

    return results


def run_one_frame(diffusion, model, clip_model, clip_vision, args, batchNum, frame_num, pbar,
                  midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization):
    global stop_on_next_loop

    # display.clear_output(wait=True)

    # Print Frame progress if animation mode is on
    # if args.animation_mode != "None":
    #     batchBar = tqdm(range(args.max_frames), desc="Frames")
    #     batchBar.n = frame_num
    #     batchBar.refresh()

    # Inits if not video frames
    if args.animation_mode != "Video Input":
        if args.init_image in ['', 'none', 'None', 'NONE']:
            init_image = None
        else:
            init_image = args.init_image
        init_scale = args.init_scale
        skip_steps = args.skip_steps

    if args.animation_mode == "2D":
        if args.key_frames:
            angle = args.angle_series[frame_num]
            zoom = args.zoom_series[frame_num]
            translation_x = args.translation_x_series[frame_num]
            translation_y = args.translation_y_series[frame_num]
            print(
                f'angle: {angle}',
                f'zoom: {zoom}',
                f'translation_x: {translation_x}',
                f'translation_y: {translation_y}',
            )

        if frame_num > 0:
            args.seed += 1
            if args.resume_run and frame_num == args.start_frame:
                img_0 = cv2.imread(
                    args.batchFolder+f"/{args.batch_name}({batchNum})_{args.start_frame-1:04}.png")
            else:
                img_0 = cv2.imread('prevFrame.png')
            center = (1*img_0.shape[1]//2, 1*img_0.shape[0]//2)
            trans_mat = np.float32(
                [[1, 0, translation_x],
                    [0, 1, translation_y]]
            )
            rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
            trans_mat = np.vstack([trans_mat, [0, 0, 1]])
            rot_mat = np.vstack([rot_mat, [0, 0, 1]])
            transformation_matrix = np.matmul(rot_mat, trans_mat)
            img_0 = cv2.warpPerspective(
                img_0,
                transformation_matrix,
                (img_0.shape[1], img_0.shape[0]),
                borderMode=cv2.BORDER_WRAP
            )

            cv2.imwrite('prevFrameScaled.png', img_0)
            init_image = 'prevFrameScaled.png'
            init_scale = args.frames_scale
            skip_steps = args.calc_frames_skip_steps

    if args.animation_mode == "3D":
        if frame_num > 0:
            args.seed += 1
            if args.resume_run and frame_num == args.start_frame:
                img_filepath = args.batchFolder + \
                    f"/{args.batch_name}({batchNum})_{args.start_frame-1:04}.png"
                if args.turbo_mode and frame_num > args.turbo_preroll:
                    shutil.copyfile(img_filepath, 'oldFrameScaled.png')
            else:
                img_filepath = 'prevFrame.png'

            next_step_pil = do_3d_step(
                args, img_filepath, frame_num, midas_model, midas_transform)
            next_step_pil.save('prevFrameScaled.png')

            # Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
            if args.turbo_mode:
                if frame_num == args.turbo_preroll:  # start tracking oldframe
                    # stash for later blending
                    next_step_pil.save('oldFrameScaled.png')
                elif frame_num > args.turbo_preroll:
                    # set up 2 warped image sequences, old & new, to blend toward new diff image
                    old_frame = do_3d_step(
                        args, 'oldFrameScaled.png', frame_num, midas_model, midas_transform)
                    old_frame.save('oldFrameScaled.png')
                    if frame_num % int(args.turbo_steps) != 0:
                        print(
                            'turbo skip this frame: skipping clip diffusion steps')
                        filename = f'{args.batch_name}({batchNum})_{frame_num:04}.png'
                        blend_factor = (
                            (frame_num % int(args.turbo_steps))+1)/int(args.turbo_steps)
                        print(
                            'turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                        # this is already updated..
                        newWarpedImg = cv2.imread('prevFrameScaled.png')
                        oldWarpedImg = cv2.imread('oldFrameScaled.png')
                        blendedImage = cv2.addWeighted(
                            newWarpedImg, blend_factor, oldWarpedImg, 1-blend_factor, 0.0)
                        cv2.imwrite(
                            f'{args.batchFolder}/{filename}', blendedImage)
                        # save it also as prev_frame to feed next iteration
                        next_step_pil.save(f'{img_filepath}')
                        if args.vr_mode:
                            generate_eye_views(
                                TRANSLATION_SCALE, args.batchFolder, filename, frame_num, midas_model, midas_transform)
                        return []
                    else:
                        # if not a skip frame, will run diffusion and need to blend.
                        oldWarpedImg = cv2.imread('prevFrameScaled.png')
                        # swap in for blending later
                        cv2.imwrite(f'oldFrameScaled.png', oldWarpedImg)
                        print('clip/diff this frame - generate clip diff image')

            init_image = 'prevFrameScaled.png'
            init_scale = args.frames_scale
            skip_steps = args.calc_frames_skip_steps

    if args.animation_mode == "Video Input":
        init_scale = args.video_init_frames_scale
        skip_steps = args.calc_frames_skip_steps
        if not args.video_init_seed_continuity:
            args.seed += 1
        if args.video_init_flow_warp:
            if frame_num == 0:
                skip_steps = args.video_init_skip_steps
                init_image = f'{args.videoFramesFolder}/{frame_num+1:04}.jpg'
            if frame_num > 0:
                prev = PIL.Image.open(
                    args.batchFolder+f"/{args.batch_name}({batchNum})_{frame_num-1:04}.png")

                frame1_path = f'{args.videoFramesFolder}/{frame_num:04}.jpg'
                frame2 = PIL.Image.open(
                    f'{args.videoFramesFolder}/{frame_num+1:04}.jpg')
                flo_path = f"/{args.flo_folder}/{frame1_path.split('/')[-1]}.npy"

                init_image = 'warped.png'
                print(args.video_init_flow_blend)
                weights_path = None
                if args.video_init_check_consistency:
                    # TBD
                    pass

                import video_input
                video_input.warp(prev, frame2, flo_path, blend=args.video_init_flow_blend,
                                    weights_path=weights_path).save(init_image)

        else:
            init_image = f'{args.videoFramesFolder}/{frame_num+1:04}.jpg'

    loss_values = []

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    target_embeds, weights = [], []

    if args.prompts_series is not None and frame_num >= len(args.prompts_series):
        frame_prompt = args.prompts_series[-1]
    elif args.prompts_series is not None:
        frame_prompt = args.prompts_series[frame_num]
    else:
        frame_prompt = []

    print(args.image_prompts_series)
    if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
        image_prompt = args.image_prompts_series[-1]
    elif args.image_prompts_series is not None:
        image_prompt = args.image_prompts_series[frame_num]
    else:
        image_prompt = []

    device = comfy.model_management.get_torch_device()
    
    print(f'Frame {frame_num} Prompt: {frame_prompt}')

    clip_models = clip_model

    model_stats = []
    for clip_model in clip_models:
    
        clip_vision = clip_model
        if isinstance(clip_model, ClipVisionModel):
            clip_vision = clip_model.model.to(device) # Gets loaded to CPU by comfy, move to GPU
    
        cutn = args.cutn
        model_stat = {"clip_model": None, "target_embeds": [],
                        "make_cutouts": None, "weights": []}
        model_stat["clip_model"] = clip_model
        model_stat["clip_vision_model"] = clip_vision

        for prompts in frame_prompt:
            for prompt in prompts:
                txt, weight = disco_utils.parse_prompt(prompt)
                txt = encode_text(clip_model, prompt).to(device)

                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append(
                            (txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0, 1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

        if image_prompt:
            input_res = get_input_resolution(clip_vision)
            model_stat["make_cutouts"] = MakeCutouts(input_res, cutn, skip_augs=args.skip_augs)
            for prompt in image_prompt:
                path, weight = disco_utils.parse_prompt(prompt)
                img = Image.open(disco_utils.fetch(path)).convert('RGB')
                img = TF.resize(
                    img, min(args.side_x, args.side_y, *img.size), T.InterpolationMode.LANCZOS)
                batch = model_stat["make_cutouts"](TF.to_tensor(
                    img).to(device).unsqueeze(0).mul(2).sub(1))
                embed = encode_images(clip_vision, disco_utils.normalize(batch))
                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append(
                            (embed + torch.randn(embed.shape).cuda() * args.rand_mag).clamp(0, 1))
                        weights.extend([weight / cutn] * cutn)
                else:
                    model_stat["target_embeds"].append(embed)
                    model_stat["weights"].extend([weight / cutn] * cutn)

        model_stat["target_embeds"] = torch.cat(
            model_stat["target_embeds"])
        model_stat["weights"] = torch.tensor(
            model_stat["weights"], device=device)
        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)

    init = None
    if init_image is not None:
        init = Image.open(disco_utils.fetch(init_image)).convert('RGB')
        init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    if args.perlin_init:
        init = disco_utils.regen_perlin(args.perlin_mode, args.batch_size)

    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]
            if args.MS.use_secondary_model is True:
                alpha = torch.tensor(
                    diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                sigma = torch.tensor(
                    diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                cosine_t = disco_utils.alpha_sigma_to_t(alpha, sigma)
                out = args.MS.secondary_model(
                    x, cosine_t[None].repeat([n])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            else:
                with torch.inference_mode(False):
                    my_t = torch.ones([n], device=device,
                                        dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(
                        model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out['pred_xstart'] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
            for model_stat in model_stats:
                for i in range(args.cutn_batches):
                    # errors on last step without +1, need to find source
                    t_int = int(t.item())+1
                    input_resolution = get_input_resolution(model_stat["clip_vision_model"])

                    cuts = MakeCutoutsDango(animation_mode=args.animation_mode,
                                            skip_augs=args.skip_augs,
                                            cut_size=input_resolution,
                                            Overview=args.cut_overview[1000-t_int],
                                            InnerCrop=args.cut_innercut[1000-t_int],
                                            IC_Size_Pow=args.cut_ic_pow[1000-t_int],
                                            IC_Grey_P=args.cut_icgray_p[1000-t_int]
                                            )
                    clip_in = disco_utils.normalize(
                        cuts(x_in.add(1).div(2)))
                    image_embeds = encode_images(model_stat["clip_vision_model"], clip_in)
                    dists = disco_utils.spherical_dist_loss(image_embeds.unsqueeze(
                        1), model_stat["target_embeds"].unsqueeze(0))
                    dists = dists.view(
                        [args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
                    losses = dists.mul(
                        model_stat["weights"]).sum(2).mean(0)
                    # log loss, probably shouldn't do per cutn_batch
                    loss_values.append(losses.sum().item())
                    grads = torch.autograd.grad(losses.sum() * args.clip_guidance_scale, x_in)
                    x_in_grad += grads[0] / args.cutn_batches
            tv_losses = disco_utils.tv_loss(x_in)
            if args.MS.use_secondary_model is True:
                range_losses = disco_utils.range_loss(out)
            else:
                with torch.inference_mode(False):
                    range_losses = disco_utils.range_loss(out['pred_xstart'])
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = tv_losses.sum() * args.tv_scale + range_losses.sum() * \
                args.range_scale + sat_losses.sum() * args.sat_scale
            if init is not None and init_scale:
                init_losses = args.MS.lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if torch.isnan(x_in_grad).any() == False:
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                # print("NaN'd")
                x_is_NaN = True
                grad = torch.zeros_like(x)
        if args.clamp_grad and x_is_NaN == False:
            magnitude = grad.square().mean().sqrt()
            # min=-0.02, min=-clamp_max,
            return grad * magnitude.clamp(max=args.clamp_max) / magnitude
        return grad

    if args.MS.diffusion_sampling_mode == 'ddim':
        sample_fn = diffusion.ddim_sample_loop_progressive
    elif args.MS.diffusion_sampling_mode == 'stsp':
        sample_fn = diffusion.stsp_sample_loop_progressive
    elif args.MS.diffusion_sampling_mode == 'ltsp':
        sample_fn = diffusion.ltsp_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    results = []

    # image_display = Output()
    for i in range(args.n_batches):
        # if args.animation_mode == 'None':
            # display.clear_output(wait=True)
            # batchBar = tqdm(range(args.n_batches), desc="Batches")
            # batchBar.n = i
            # batchBar.refresh()
        # display.display(image_display)
        gc.collect()
        torch.cuda.empty_cache()
        cur_t = diffusion.num_timesteps - skip_steps - 1
        total_steps = cur_t

        if args.perlin_init:
            init = disco_utils.regen_perlin(
                args.perlin_mode, args.batch_size, True)

        symmetry_transformation_fn = id
        if args.use_horizontal_symmetry:
            symmetry_transformation_fn = horiz_symmetry
        if args.use_vertical_symmetry:
            symmetry_transformation_fn = vert_symmetry

        if args.MS.diffusion_sampling_mode == 'ddim':
            samples = sample_fn(
                model,
                (args.batch_size, 3, args.side_y, args.side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                eta=args.eta,
                transformation_fn=symmetry_transformation_fn,
                transformation_percent=args.transformation_percent
            )
        else:
            samples = sample_fn(
                model,
                (args.batch_size, 3, args.side_y, args.side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                order=2,
            )

        # with run_display:
        # display.clear_output(wait=True)
        for j, sample in enumerate(samples):
            pbar.update_absolute(j, diffusion.num_timesteps - skip_steps)
            cur_t -= 1
            intermediateStep = False
            if args.steps_per_checkpoint is not None:
                if j % args.steps_per_checkpoint == 0 and j > 0:
                    intermediateStep = True
            elif j in args.intermediate_saves:
                intermediateStep = True
            # with image_display:
            if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                for k, image in enumerate(sample['pred_xstart']):
                    # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                    datetime.now().strftime('%y%m%d-%H%M%S_%f')
                    percent = math.ceil(j/total_steps*100)
                    # if args.n_batches > 0:
                    #     # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                    #     if cur_t == -1 and args.intermediates_in_subfolder is True:
                    #         save_num = f'{frame_num:04}' if args.animation_mode != "None" else i
                    #         filename = f'{args.batch_name}({batchNum})_{save_num}.png'
                    #     else:
                    #         # If we're working with percentages, append it
                    #         if args.steps_per_checkpoint is not None:
                    #             filename = f'{args.batch_name}({batchNum})_{i:04}-{percent:02}%.png'
                    #         # Or else, iIf we're working with specific steps, append those
                    #         else:
                    #             filename = f'{args.batch_name}({batchNum})_{i:04}-{j:03}.png'
                    # save_image(image, j, cur_t, filename, frame_num, midas_model, midas_transform, args)

                    if cur_t == -1:
                        # We get back a tensor of size [C, H, W].
                        # Comfy's IMAGE output expects a stacked [B, H, W, C].
                        # So... Let's Transposing!
                        image = image.permute(1, 2, 0).add(1).div(2).clamp(0, 1)
                        # image = image.add(1).div(2).clamp(0, 1)

                        # [[H, W, C]] -> [B, H, W, C]
                        # All results will be wrapped in a Python list for use with OUTPUT_IS_LIST.
                        # B will always be 1 for each individual image tensor.
                        # Yes this is weird.
                        results.append(torch.stack([image]))

        # plt.plot(np.array(loss_values), 'r')

    return results

def save_image(image, j, cur_t, filename, frame_num, midas_model, midas_transform, args):
    image = TF.to_pil_image(
        image.add(1).div(2).clamp(0, 1))
    # if j % args.display_rate == 0 or cur_t == -1:
        # image.save('progress.png')
        # display.clear_output(wait=True)
        # display.display(display.Image('progress.png'))
    if args.steps_per_checkpoint is not None:
        if j % args.steps_per_checkpoint == 0 and j > 0:
            if args.intermediates_in_subfolder is True:
                image.save(
                    f'{args.partialFolder}/{filename}')
            else:
                image.save(
                    f'{args.batchFolder}/{filename}')
    else:
        if j in args.intermediate_saves:
            if args.intermediates_in_subfolder is True:
                image.save(
                    f'{args.partialFolder}/{filename}')
            else:
                image.save(
                    f'{args.batchFolder}/{filename}')
    if cur_t == -1:
        # if frame_num == 0:
        #   save_settings()
        if args.animation_mode != "None":
            image.save('prevFrame.png')
        image.save(f'{args.batchFolder}/{filename}')
        if args.animation_mode == "3D":
            # If turbo, save a blended image
            if args.turbo_mode and frame_num > 0:
                # Mix new image with prevFrameScaled
                blend_factor = (1)/int(args.turbo_steps)
                # This is already updated..
                newFrame = cv2.imread('prevFrame.png')
                prev_frame_warped = cv2.imread(
                    'prevFrameScaled.png')
                blendedImage = cv2.addWeighted(
                    newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                cv2.imwrite(
                    f'{args.batchFolder}/{filename}', blendedImage)
            else:
                image.save(
                    f'{args.batchFolder}/{filename}')

            if args.vr_mode:
                generate_eye_views(
                    args, TRANSLATION_SCALE, args.batchFolder, filename, frame_num, midas_model, midas_transform)

        # if frame_num != args.max_frames-1:
        #   display.clear_output()


def generate_eye_views(args, trans_scale, batchFolder, filename, frame_num, midas_model, midas_transform):
    device = comfy.model_management.get_torch_device()
    for i in range(2):
        theta = args.vr_eye_angle * (math.pi/180)
        ray_origin = math.cos(theta) * args.vr_ipd / \
            2 * (-1.0 if i == 0 else 1.0)
        ray_rotation = (theta if i == 0 else -theta)
        translate_xyz = [-(ray_origin)*trans_scale, 0, 0]
        rotate_xyz = [0, (ray_rotation), 0]
        rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(
            rotate_xyz, device=device), "XYZ").unsqueeze(0)
        transformed_image = dxf.transform_image_3d(f'{batchFolder}/{filename}', midas_model, midas_transform, device,
                                                   rot_mat, translate_xyz, args.near_plane, args.far_plane,
                                                   args.fov, padding_mode=args.padding_mode,
                                                   sampling_mode=args.sampling_mode, midas_weight=args.midas_weight, spherical=True)
        eye_file_path = batchFolder + \
            f"/frame_{frame_num:04}" + ('_l' if i == 0 else '_r')+'.png'
        transformed_image.save(eye_file_path)

# def save_settings():
#     setting_list = {
#       'text_prompts': text_prompts,
#       'image_prompts': image_prompts,
#       'clip_guidance_scale': clip_guidance_scale,
#       'tv_scale': tv_scale,
#       'range_scale': range_scale,
#       'sat_scale': sat_scale,
#       # 'cutn': cutn,
#       'cutn_batches': cutn_batches,
#       'max_frames': max_frames,
#       'interp_spline': interp_spline,
#       # 'rotation_per_frame': rotation_per_frame,
#       'init_image': init_image,
#       'init_scale': init_scale,
#       'skip_steps': skip_steps,
#       # 'zoom_per_frame': zoom_per_frame,
#       'frames_scale': frames_scale,
#       'frames_skip_steps': frames_skip_steps,
#       'perlin_init': perlin_init,
#       'perlin_mode': perlin_mode,
#       'skip_augs': skip_augs,
#       'randomize_class': randomize_class,
#       'clip_denoised': clip_denoised,
#       'clamp_grad': clamp_grad,
#       'clamp_max': clamp_max,
#       'seed': seed,
#       'fuzzy_prompt': fuzzy_prompt,
#       'rand_mag': rand_mag,
#       'eta': eta,
#       'width': width_height[0],
#       'height': width_height[1],
#       'diffusion_model': diffusion_model,
#       'use_secondary_model': use_secondary_model,
#       'steps': steps,
#       'diffusion_steps': diffusion_steps,
#       'diffusion_sampling_mode': diffusion_sampling_mode,
#       'ViTB32': ViTB32,
#       'ViTB16': ViTB16,
#       'ViTL14': ViTL14,
#       'ViTL14_336px': ViTL14_336px,
#       'RN101': RN101,
#       'RN50': RN50,
#       'RN50x4': RN50x4,
#       'RN50x16': RN50x16,
#       'RN50x64': RN50x64,
#       'ViTB32_laion2b_e16': ViTB32_laion2b_e16,
#       'ViTB32_laion400m_e31': ViTB32_laion400m_e31,
#       'ViTB32_laion400m_32': ViTB32_laion400m_32,
#       'ViTB32quickgelu_laion400m_e31': ViTB32quickgelu_laion400m_e31,
#       'ViTB32quickgelu_laion400m_e32': ViTB32quickgelu_laion400m_e32,
#       'ViTB16_laion400m_e31': ViTB16_laion400m_e31,
#       'ViTB16_laion400m_e32': ViTB16_laion400m_e32,
#       'RN50_yffcc15m': RN50_yffcc15m,
#       'RN50_cc12m': RN50_cc12m,
#       'RN50_quickgelu_yfcc15m': RN50_quickgelu_yfcc15m,
#       'RN50_quickgelu_cc12m': RN50_quickgelu_cc12m,
#       'RN101_yfcc15m': RN101_yfcc15m,
#       'RN101_quickgelu_yfcc15m': RN101_quickgelu_yfcc15m,
#       'cut_overview': str(cut_overview),
#       'cut_innercut': str(cut_innercut),
#       'cut_ic_pow': str(cut_ic_pow),
#       'cut_icgray_p': str(cut_icgray_p),
#       'key_frames': key_frames,
#       'max_frames': max_frames,
#       'angle': angle,
#       'zoom': zoom,
#       'translation_x': translation_x,
#       'translation_y': translation_y,
#       'translation_z': translation_z,
#       'rotation_3d_x': rotation_3d_x,
#       'rotation_3d_y': rotation_3d_y,
#       'rotation_3d_z': rotation_3d_z,
#       'midas_depth_model': midas_depth_model,
#       'midas_weight': midas_weight,
#       'near_plane': near_plane,
#       'far_plane': far_plane,
#       'fov': fov,
#       'padding_mode': padding_mode,
#       'sampling_mode': sampling_mode,
#       'video_init_path':video_init_path,
#       'extract_nth_frame':extract_nth_frame,
#       'video_init_seed_continuity': video_init_seed_continuity,
#       'turbo_mode':turbo_mode,
#       'turbo_steps':turbo_steps,
#       'turbo_preroll':turbo_preroll,
#       'use_horizontal_symmetry':use_horizontal_symmetry,
#       'use_vertical_symmetry':use_vertical_symmetry,
#       'transformation_percent':transformation_percent,
#       #video init settings
#       'video_init_steps': video_init_steps,
#       'video_init_clip_guidance_scale': video_init_clip_guidance_scale,
#       'video_init_tv_scale': video_init_tv_scale,
#       'video_init_range_scale': video_init_range_scale,
#       'video_init_sat_scale': video_init_sat_scale,
#       'video_init_cutn_batches': video_init_cutn_batches,
#       'video_init_skip_steps': video_init_skip_steps,
#       'video_init_frames_scale': video_init_frames_scale,
#       'video_init_frames_skip_steps': video_init_frames_skip_steps,
#       #warp settings
#       'video_init_flow_warp':video_init_flow_warp,
#       'video_init_flow_blend':video_init_flow_blend,
#       'video_init_check_consistency':video_init_check_consistency,
#       'video_init_blend_mode':video_init_blend_mode
#     }
#     # print('Settings:', setting_list)
#     with open(f"{batchFolder}/{batch_name}({batchNum})_settings.txt", "w+", encoding="utf-8") as f:   #save settings
#         json.dump(setting_list, f, ensure_ascii=False, indent=4)
