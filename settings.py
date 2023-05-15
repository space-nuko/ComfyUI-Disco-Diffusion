from dataclasses import dataclass
import pathlib
import os
import pathlib
import os
from dataclasses import dataclass
import cv2
import pandas as pd
import gc
import math
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import numpy as np
from numpy import asarray
import subprocess

from .video_input import setup_raft
from .video_input import setup_video_input_mode
from .video_input import generate_optical_flow
from .model_settings import ModelSettings


@dataclass
class DiscoDiffusionSettings:
    def __init__(self):
        self.root_path = os.getcwd()
        self.initDirPath = f'{self.root_path}/init_images'
        os.makedirs(self.initDirPath, exist_ok=True)
        self.outDirPath = f'{self.root_path}/images_out'
        os.makedirs(self.outDirPath, exist_ok=True)

        self.useCPU = False

        # %%
        # !! {"metadata":{
        # !!   "id": "BasicSettings"
        # !! }}
        # @markdown ####**Basic Settings:**
        self.batch_name = 'TimeToDisco'  # @param{type: 'string'}
        self.batch_size = 1
        self.n_batches = 1
        # @param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
        self.steps = 250
        self.width_height_for_512x512_models = [
            1280, 768]  # @param{type: 'raw'}
        self.clip_guidance_scale = 5000  # @param{type: 'number'}
        self.tv_scale = 0  # @param{type: 'number'}
        self.range_scale = 150  # @param{type: 'number'}
        self.sat_scale = 0  # @param{type: 'number'}
        self.cutn = 16
        self.cutn_batches = 4  # @param{type: 'number'}
        self.skip_augs = False  # @param{type: 'boolean'}

        # @markdown ####**Image dimensions to be used for 256x256 models (e.g. pixelart models):**
        self.width_height_for_256x256_models = [
            512, 448]  # @param{type: 'raw'}

        # @markdown ####**Video Init Basic Settings:**
        # @param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
        self.video_init_steps = 100
        self.video_init_clip_guidance_scale = 1000  # @param{type: 'number'}
        self.video_init_tv_scale = 0.1  # @param{type: 'number'}
        self.video_init_range_scale = 150  # @param{type: 'number'}
        self.video_init_sat_scale = 300  # @param{type: 'number'}
        self.video_init_cutn_batches = 4  # @param{type: 'number'}
        self.video_init_skip_steps = 50  # @param{type: 'integer'}

        # @markdown ---

        # @markdown ####**Init Image Settings:**
        self.init_image = None  # @param{type: 'string'}
        self.init_scale = 1000  # @param{type: 'integer'}
        self.skip_steps = 10  # @param{type: 'integer'}
        # @markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.*

        # Make folder for batch
        self.batchFolder = f'{self.outDirPath}/{self.batch_name}'
        os.makedirs(self.batchFolder, exist_ok=True)

        # @markdown ####**Animation Mode:**
        # @param ['None', '2D', '3D', 'Video Input'] {type:'string'}
        self.animation_mode = 'None'
        # @markdown *For animation, you probably want to turn `cutn_batches` to 1 to make it quicker.*

        self.video_init_path = "init.mp4"  # @param {type: 'string'}
        self.extract_nth_frame = 2  # @param {type: 'number'}
        # @param {type: 'boolean'}
        self.persistent_frame_output_in_batch_folder = True
        self.video_init_seed_continuity = False  # @param {type: 'boolean'}
        # @markdown #####**Video Optical Flow Settings:**
        self.video_init_flow_warp = True  # @param {type: 'boolean'}
        # Call optical flow from video frames and warp prev frame with flow
        # @param {type: 'number'} #0 - take next frame, 1 - take prev warped frame
        self.video_init_flow_blend = 0.999
        self.video_init_check_consistency = False  # Insert param here when ready
        # @param ['None', 'linear', 'optical flow']
        self.video_init_blend_mode = "optical flow"

        # @markdown ---

        # @markdown ####**2D Animation Settings:**
        # @markdown `zoom` is a multiplier of dimensions, 1 is no zoom.
        # @markdown All rotations are provided in degrees.

        self.key_frames = True  # @param {type:"boolean"}
        self.max_frames = 10  # @param {type:"number"}

        # Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
        self.interp_spline = 'Linear'
        self.angle = "0:(0)"  # @param {type:"string"}
        self.zoom = "0: (1), 10: (1.05)"  # @param {type:"string"}
        self.translation_x = "0: (0)"  # @param {type:"string"}
        self.translation_y = "0: (0)"  # @param {type:"string"}
        self.translation_z = "0: (10.0)"  # @param {type:"string"}
        self.rotation_3d_x = "0: (0)"  # @param {type:"string"}
        self.rotation_3d_y = "0: (0)"  # @param {type:"string"}
        self.rotation_3d_z = "0: (0)"  # @param {type:"string"}
        self.midas_depth_model = "dpt_large"  # @param {type:"string"}
        self.midas_weight = 0.3  # @param {type:"number"}
        self.near_plane = 200  # @param {type:"number"}
        self.far_plane = 10000  # @param {type:"number"}
        self.fov = 40  # @param {type:"number"}
        self.padding_mode = 'border'  # @param {type:"string"}
        self.sampling_mode = 'bicubic'  # @param {type:"string"}

        # ======= TURBO MODE
        # @markdown ---
        # @markdown ####**Turbo Mode (3D anim only):**
        # @markdown (Starts after frame 10,) skips diffusion steps and just uses depth map to warp images for skipped frames.
        # @markdown Speeds up rendering by 2x-4x, and may improve image coherence between frames.
        # @markdown For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

        self.turbo_mode = False  # @param {type:"boolean"}
        self.turbo_steps = "3"  # @param ["2","3","4","5","6"] {type:"string"}
        self.turbo_preroll = 10  # frames

        # insist turbo be used only w 3d anim.
        if self.turbo_mode and self.animation_mode != '3D':
            print('=====')
            print('Turbo mode only available with 3D animations. Disabling Turbo.')
            print('=====')
            self.turbo_mode = False

        # @markdown ---

        # @markdown ####**Coherency Settings:**
        # @markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
        self.frames_scale = 1500  # @param{type: 'integer'}
        # @markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
        # @param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}
        self.frames_skip_steps = '60%'

        # @markdown ####**Video Init Coherency Settings:**
        # @markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
        self.video_init_frames_scale = 15000  # @param{type: 'integer'}
        # @markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
        # @param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}
        self.video_init_frames_skip_steps = '70%'

        # ======= VR MODE
        # @markdown ---
        # @markdown ####**VR Mode (3D anim only):**
        # @markdown Enables stereo rendering of left/right eye views (supporting Turbo) which use a different (fish-eye) camera projection matrix.
        # @markdown Note the images you're prompting will work better if they have some inherent wide-angle aspect
        # @markdown The generated images will need to be combined into left/right videos. These can then be stitched into the VR180 format.
        # @markdown Google made the VR180 Creator tool but subsequently stopped supporting it. It's available for download in a few places including https://www.patrickgrunwald.de/vr180-creator-download
        # @markdown The tool is not only good for stitching (videos and photos) but also for adding the correct metadata into existing videos, which is needed for services like YouTube to identify the format correctly.
        # @markdown Watching YouTube VR videos isn't necessarily the easiest depending on your headset. For instance Oculus have a dedicated media studio and store which makes the files easier to access on a Quest https://creator.oculus.com/manage/mediastudio/
        # @markdown
        # @markdown The command to get ffmpeg to concat your frames for each eye is in the form: `ffmpeg -framerate 15 -i frame_%4d_l.png l.mp4` (repeat for r)

        self.vr_mode = False  # @param {type:"boolean"}
        # @markdown `vr_eye_angle` is the y-axis rotation of the eyes towards the center
        self.vr_eye_angle = 0.5  # @param{type:"number"}
        # @markdown interpupillary distance (between the eyes)
        self.vr_ipd = 5.0  # @param{type:"number"}

        # %%
        # !! {"metadata":{
        # !!   "id": "ExtraSetTop"
        # !! }}
        # """
        # ### Extra Settings
        # Partial Saves, Advanced Settings, Cutn Scheduling
        # """

        # %%
        # !! {"metadata":{
        # !!   "id": "ExtraSettings"
        # !! }}
        # @markdown ####**Saving:**

        self.intermediate_saves = 0  # @param{type: 'raw'}
        self.intermediates_in_subfolder = True  # @param{type: 'boolean'}
        # @markdown Intermediate steps will save a copy at your specified intervals. You can either format it as a single integer or a list of specific steps

        # @markdown A value of `2` will save a copy at 33% and 66%. 0 will save none.

        # @markdown A value of `[5, 9, 34, 45]` will save at steps 5, 9, 34, and 45. (Make sure to include the brackets)

        # @markdown ---

        # @markdown ####**Advanced Settings:**
        # @markdown *There are a few extra advanced settings available if you double click this cell.*

        # @markdown *Perlin init will replace your init, so uncheck if using one.*

        self.perlin_init = False  # @param{type: 'boolean'}
        self.perlin_mode = 'mixed'  # @param ['mixed', 'color', 'gray']
        self.seed = 0
        self.set_seed = 'random_seed'  # @param{type: 'string'}
        self.eta = 0.8  # @param{type: 'number'}
        self.clamp_grad = True  # @param{type: 'boolean'}
        self.clamp_max = 0.05  # @param{type: 'number'}

        # EXTRA ADVANCED SETTINGS:
        self.randomize_class = True
        self.clip_denoised = False
        self.fuzzy_prompt = False
        self.rand_mag = 0.05

        # @markdown ---

        # @markdown ####**Cutn Scheduling:**
        # @markdown Format: `[40]*400+[20]*600` = 40 cuts for the first 400 /1000 steps, then 20 for the last 600/1000

        # @markdown cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.

        self.cut_overview = "[12]*400+[4]*600"  # @param {type: 'string'}
        self.cut_innercut = "[4]*400+[12]*600"  # @param {type: 'string'}
        self.cut_ic_pow = "[1]*1000"  # @param {type: 'string'}
        self.cut_icgray_p = "[0.2]*400+[0]*600"  # @param {type: 'string'}

        # @markdown KaliYuga model settings. Refer to [cut_ic_pow](https://ezcharts.miraheze.org/wiki/Category:Cut_ic_pow) as a guide. Values between 1 and 100 all work.
        # @param {type: 'string'}
        self.pad_or_pulp_cut_overview = "[15]*100+[15]*100+[12]*100+[12]*100+[6]*100+[4]*100+[2]*200+[0]*200"
        # @param {type: 'string'}
        self.pad_or_pulp_cut_innercut = "[1]*100+[1]*100+[4]*100+[4]*100+[8]*100+[8]*100+[10]*200+[10]*200"
        # @param {type: 'string'}
        self.pad_or_pulp_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300"
        # @param {type: 'string'}
        self.pad_or_pulp_cut_icgray_p = "[0.87]*100+[0.78]*50+[0.73]*50+[0.64]*60+[0.56]*40+[0.50]*50+[0.33]*100+[0.19]*150+[0]*400"

        # @param {type: 'string'}
        self.watercolor_cut_overview = "[14]*200+[12]*200+[4]*400+[0]*200"
        # @param {type: 'string'}
        self.watercolor_cut_innercut = "[2]*200+[4]*200+[12]*400+[12]*200"
        # @param {type: 'string'}
        self.watercolor_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300"
        # @param {type: 'string'}
        self.watercolor_cut_icgray_p = "[0.7]*100+[0.6]*100+[0.45]*100+[0.3]*100+[0]*600"

        # @markdown ---

        # @markdown ####**Transformation Settings:**
        self.use_vertical_symmetry = False  # @param {type:"boolean"}
        self.use_horizontal_symmetry = False  # @param {type:"boolean"}
        self.transformation_percent = [0.09]  # @param

        # %%
        # !! {"metadata":{
        # !!   "id": "PromptsTop"
        # !! }}
        """
        ### Prompts
        `animation_mode: None` will only use the first set. `animation_mode: 2D / Video` will run through them per the set frames and hold on the last one.
        """

        # %%
        # !! {"metadata":{
        # !!   "id": "Prompts"
        # !! }}
        # Note: If using a pixelart diffusion model, try adding "#pixelart" to the end of the prompt for a stronger effect. It'll tend to work a lot better!
        self.text_prompts = {
            0: ["A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.", "yellow color scheme"]
            # 100: ["This set of prompts start at frame 100", "This prompt has weight five:5"],
        }

        self.image_prompts = {
            # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
        }

    def setup(self, MS: ModelSettings):
        self.MS = MS

        if type(self.intermediate_saves) is not list:
            if self.intermediate_saves:
                self.steps_per_checkpoint = math.floor(
                    (self.steps - self.skip_steps - 1) // (self.intermediate_saves+1))
                self.steps_per_checkpoint = self.steps_per_checkpoint if self.steps_per_checkpoint > 0 else 1
                print(f'Will save every {self.steps_per_checkpoint} steps')
            else:
                self.steps_per_checkpoint = self.steps+10
        else:
            self.steps_per_checkpoint = None

        if self.intermediate_saves and self.intermediates_in_subfolder is True:
            self.partialFolder = f'{self.batchFolder}/partials'
            os.makedirs(self.partialFolder, exist_ok=True)

        self.width_height = self.width_height_for_256x256_models if MS.diffusion_model in MS.diffusion_models_256x256_list else self.width_height_for_512x512_models

        # Get corrected sizes
        self.side_x = (self.width_height[0]//64)*64
        self.side_y = (self.width_height[1]//64)*64
        if self.side_x != self.width_height[0] or self.side_y != self.width_height[1]:
            print(
                f'Changing output size to {self.side_x}x{self.side_y}. Dimensions must by multiples of 64.')

        if (MS.diffusion_model in MS.kaliyuga_pixel_art_model_names) or (MS.diffusion_model in MS.kaliyuga_pulpscifi_model_names):
            self.cut_overview = self.pad_or_pulp_cut_overview
            self.cut_innercut = self.pad_or_pulp_cut_innercut
            self.cut_ic_pow = self.pad_or_pulp_cut_ic_pow
            self.cut_icgray_p = self.pad_or_pulp_cut_icgray_p
        elif MS.diffusion_model in MS.kaliyuga_watercolor_model_names:
            self.cut_overview = self.watercolor_cut_overview
            self.cut_innercut = self.watercolor_cut_innercut
            self.cut_ic_pow = self.watercolor_cut_ic_pow
            self.cut_icgray_p = self.watercolor_cut_icgray_p

        # insist VR be used only w 3d anim.
        if self.vr_mode and self.animation_mode != '3D':
            print('=====')
            print('VR mode only available with 3D animations. Disabling VR.')
            print('=====')
            self.vr_mode = False

        if self.key_frames:
            try:
                self.angle_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.angle))
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `angle` correctly for key frames.\n"
                    "Attempting to interpret `angle` as "
                    f'"0: ({self.angle})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                self.angle = f"0: ({self.angle})"
                self.angle_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.angle))

            try:
                self.zoom_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.zoom))
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `zoom` correctly for key frames.\n"
                    "Attempting to interpret `zoom` as "
                    f'"0: ({self.zoom})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                self.zoom = f"0: ({self.zoom})"
                self.zoom_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.zoom))

            try:
                self.translation_x_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.translation_x))
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_x` correctly for key frames.\n"
                    "Attempting to interpret `translation_x` as "
                    f'"0: ({self.translation_x})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                self.translation_x = f"0: ({self.translation_x})"
                self.translation_x_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.translation_x))

            try:
                self.translation_y_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.translation_y))
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_y` correctly for key frames.\n"
                    "Attempting to interpret `translation_y` as "
                    f'"0: ({self.translation_y})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                self.translation_y = f"0: ({self.translation_y})"
                self.translation_y_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.translation_y))

            try:
                get_inbetweens(self.max_frames, self.interp_spline,
                               parse_key_frames(self.translation_z))
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_z` correctly for key frames.\n"
                    "Attempting to interpret `translation_z` as "
                    f'"0: ({self.translation_z})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                self.translation_z = f"0: ({self.translation_z})"
                self.translation_z_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.translation_z))

            try:
                self.rotation_3d_x_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.rotation_3d_x))
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `rotation_3d_x` correctly for key frames.\n"
                    "Attempting to interpret `rotation_3d_x` as "
                    f'"0: ({self.rotation_3d_x})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                self.rotation_3d_x = f"0: ({self.rotation_3d_x})"
                self.rotation_3d_x_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.rotation_3d_x))

            try:
                self.rotation_3d_y_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.rotation_3d_y))
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `rotation_3d_y` correctly for key frames.\n"
                    "Attempting to interpret `rotation_3d_y` as "
                    f'"0: ({self.rotation_3d_y})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                self.rotation_3d_y = f"0: ({self.rotation_3d_y})"
                self.rotation_3d_y_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.rotation_3d_y))

            try:
                self.rotation_3d_z_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.rotation_3d_z))
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `rotation_3d_z` correctly for key frames.\n"
                    "Attempting to interpret `rotation_3d_z` as "
                    f'"0: ({self.rotation_3d_z})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                self.rotation_3d_z = f"0: ({self.rotation_3d_z})"
                self.rotation_3d_z_series = get_inbetweens(
                    self.max_frames, self.interp_spline, parse_key_frames(self.rotation_3d_z))

        else:
            self.angle = float(self.angle)
            self.zoom = float(self.zoom)
            self.translation_x = float(self.translation_x)
            self.translation_y = float(self.translation_y)
            self.translation_z = float(self.translation_z)
            self.rotation_3d_x = float(self.rotation_3d_x)
            self.rotation_3d_y = float(self.rotation_3d_y)
            self.rotation_3d_z = float(self.rotation_3d_z)

        if self.animation_mode == 'Video Input':
            self.max_frames = len(glob(f'{self.videoFramesFolder}/*.jpg'))

            # Call optical flow from video frames and warp prev frame with flow
            if self.persistent_frame_output_in_batch_folder:  # suggested by Chris the Wizard#8082 at discord
                self.videoFramesFolder = f'{self.batchFolder}/videoFrames'
            else:
                self.videoFramesFolder = f'/content/videoFrames'
            os.makedirs(self.videoFramesFolder, exist_ok=True)
            print(
                f"Exporting Video Frames (1 every {self.extract_nth_frame})...")
            try:
                for f in pathlib.Path(f'{self.videoFramesFolder}').glob('*.jpg'):
                    f.unlink()
            except Exception as err:
                print(err)
            vf = f'select=not(mod(n\,{self.extract_nth_frame}))'
            if os.path.exists(self.video_init_path):
                subprocess.run(['ffmpeg', '-i', f'{self.video_init_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel',
                               'error', '-stats', f'{self.videoFramesFolder}/%04d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
            else:
                print(
                    f'\nWARNING!\n\nVideo not found: {self.video_init_path}.\nPlease check your video path.\n')
            #!ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {videoFramesFolder}/%04d.jpg

            setup_raft()
            setup_video_input_mode(self)
            generate_optical_flow(self)


def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.

    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.

    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param

    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def get_inbetweens(max_frames, interp_method, key_frames, integer=False):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.

    Parameters
    ----------
    key_frames: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.

    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.

    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64

    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'

    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames -
                     1] = key_frame_series[key_frame_series.last_valid_index()]
    # key_frame_series = key_frame_series.interpolate(method=intrp_method,order=1, limit_direction='both')
    key_frame_series = key_frame_series.interpolate(
        method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series
