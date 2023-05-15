import shutil
import sys
import cv2
import pandas as pd
import gc
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
import warnings
import PIL
from tqdm import trange
import os

from .settings import DiscoDiffusionSettings

skip_video_for_run_all = False  # @param {type: 'boolean'}

# @title ### **Create video**
# @markdown Video file will save in the same folder as your images.


def create_video(batchNum, args: DiscoDiffusionSettings):
    if args.animation_mode == 'Video Input':
        frames = sorted(glob(args.in_path+'/*.*'))
        if len(frames) == 0:
            sys.exit(
                "ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell.")
        flows = glob(args.flo_folder+'/*.*')
        if (len(flows) == 0) and args.video_init_flow_warp:
            sys.exit(
                "ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")

    blend = 0.5  # @param {type: 'number'}
    args.video_init_check_consistency = False  # @param {type: 'boolean'}
    if skip_video_for_run_all == True:
        print(
            'Skipping video creation, uncheck skip_video_for_run_all if you want to run it')

    else:
        # import subprocess in case this cell is run without the above cells
        import subprocess

        latest_run = batchNum

        folder = args.batch_name  # @param
        run = latest_run  # @param
        final_frame = 'final_frame'

        # @param {type:"number"} This is the frame where the video will start
        init_frame = 1
        # @param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
        last_frame = final_frame
        fps = 12  # @param {type:"number"}
        # view_video_in_cell = True #@param {type: 'boolean'}

        frames = []
        # tqdm.write('Generating video...')

        if last_frame == 'final_frame':
            last_frame = len(glob(args.batchFolder+f"/{folder}({run})_*.png"))
            print(f'Total frames: {last_frame}')

        image_path = f"{args.outDirPath}/{folder}/{folder}({run})_%04d.png"
        filepath = f"{args.outDirPath}/{folder}/{folder}({run}).mp4"

        if (args.video_init_blend_mode == 'optical flow') and (args.animation_mode == 'Video Input'):
            image_path = f"{args.outDirPath}/{folder}/flow/{folder}({run})_%04d.png"
            filepath = f"{args.outDirPath}/{folder}/{folder}({run})_flow.mp4"
            if last_frame == 'final_frame':
                last_frame = len(
                    glob(args.batchFolder+f"/flow/{folder}({run})_*.png"))
            flo_out = args.batchFolder+f"/flow"
            args.createPath(flo_out)
            frames_in = sorted(
                glob(args.batchFolder+f"/{folder}({run})_*.png"))
            shutil.copy(frames_in[0], flo_out)
            for i in trange(init_frame, min(len(frames_in), last_frame)):
                frame1_path = frames_in[i-1]
                frame2_path = frames_in[i]

                frame1 = PIL.Image.open(frame1_path)
                frame2 = PIL.Image.open(frame2_path)
                frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):04}.jpg"
                flo_path = f"/{args.flo_folder}/{frame1_stem}.npy"
                weights_path = None
                if args.video_init_check_consistency:
                    # TBD
                    pass
                import video_input
                video_input.warp(frame1, frame2, flo_path, blend=blend, weights_path=weights_path).save(
                    args.batchFolder+f"/flow/{folder}({run})_{i:04}.png")
        if args.video_init_blend_mode == 'linear':
            image_path = f"{args.outDirPath}/{folder}/blend/{folder}({run})_%04d.png"
            filepath = f"{args.outDirPath}/{folder}/{folder}({run})_blend.mp4"
            if last_frame == 'final_frame':
                last_frame = len(
                    glob(args.batchFolder+f"/blend/{folder}({run})_*.png"))
            blend_out = args.batchFolder+f"/blend"
            os.makedirs(blend_out, exist_ok=True)
            frames_in = glob(args.batchFolder+f"/{folder}({run})_*.png")
            shutil.copy(frames_in[0], blend_out)
            for i in trange(1, len(frames_in)):
                frame1_path = frames_in[i-1]
                frame2_path = frames_in[i]

                frame1 = PIL.Image.open(frame1_path)
                frame2 = PIL.Image.open(frame2_path)

                frame = PIL.Image.fromarray((np.array(frame1)*(1-blend) + np.array(frame2)*(
                    blend)).astype('uint8')).save(args.batchFolder+f"/blend/{folder}({run})_{i:04}.png")

        cmd = [
            'ffmpeg',
            '-y',
            '-vcodec',
            'png',
            '-r',
            str(fps),
            '-start_number',
            str(init_frame),
            '-i',
            image_path,
            '-frames:v',
            str(last_frame+1),
            '-c:v',
            'libx264',
            '-vf',
            f'fps={fps}',
            '-pix_fmt',
            'yuv420p',
            '-crf',
            '17',
            '-preset',
            'veryslow',
            filepath
        ]

        process = subprocess.Popen(
            cmd, cwd=f'{args.batchFolder}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)
        else:
            print("The video is ready and saved to the images folder")

        # if view_video_in_cell:
        #     mp4 = open(filepath,'rb').read()
        #     data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        #     display.HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>')

    # %%
    # !! {"main_metadata":{
    # !!   "anaconda-cloud": {},
    # !!   "accelerator": "GPU",
    # !!   "colab": {
    # !!     "collapsed_sections": [
    # !!       "CreditsChTop",
    # !!       "TutorialTop",
    # !!       "CheckGPU",
    # !!       "InstallDeps",
    # !!       "DefMidasFns",
    # !!       "DefFns",
    # !!       "DefSecModel",
    # !!       "DefSuperRes",
    # !!       "AnimSetTop",
    # !!       "ExtraSetTop",
    # !!       "InstallRAFT",
    # !!       "CustModel",
    # !!       "FlowFns1",
    # !!       "FlowFns2"
    # !!     ],
    # !!     "machine_shape": "hm",
    # !!     "name": "Disco Diffusion v5.61 [Now with portrait_generator_v001]",
    # !!     "private_outputs": true,
    # !!     "provenance": [],
    # !!     "include_colab_link": true
    # !!   },
    # !!   "kernelspec": {
    # !!     "display_name": "Python 3",
    # !!     "language": "python",
    # !!     "name": "python3"
    # !!   },
    # !!   "language_info": {
    # !!     "codemirror_mode": {
    # !!       "name": "ipython",
    # !!       "version": 3
    # !!     },
    # !!     "file_extension": ".py",
    # !!     "mimetype": "text/x-python",
    # !!     "name": "python",
    # !!     "nbconvert_exporter": "python",
    # !!     "pygments_lexer": "ipython3",
    # !!     "version": "3.6.1"
    # !!   }
    # !! }}
