import PIL
import argparse
from PIL import Image
import pathlib
import os
import cv2
import pandas as pd
import gc
import subprocess
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
from raft import RAFT
from raftutils.utils import InputPadder
from raftutils import flow_viz


from folder_paths import models_dir


# %%
# !! {"metadata":{
# !!   "id": "InstallRAFT"
# !! }}
# @title Install RAFT for Video input animation mode only
# @markdown Run once per session. Doesn't download again if model path exists.
# @markdown Use force download to reload raft models if needed
force_download = False  # @param {type:'boolean'}


def setup_raft():
    pass

# @title Define optical flow functions for Video input animation mode only
def setup_video_input_mode(S):
    S.in_path = S.videoFramesFolder
    # f'{in_path}/out_flo_fwd'
    # f'{models_dir}/RAFT/core'


args2 = argparse.Namespace()
args2.small = False
args2.mixed_precision = True


TAG_CHAR = np.array([202021.25], np.float32)


def writeFlow(filename, uv, v=None):
    """
    https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
    Copyright 2017 NVIDIA CORPORATION

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:, np.arange(width)*2] = u
    tmp[:, np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def load_img(img, size):
    img = Image.open(img).convert('RGB').resize(size)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()


def get_flow(frame1, frame2, model, iters=20):
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    _, flow12 = model(frame1, frame2, iters=iters, test_mode=True)
    flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

    return flow12


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def makeEven(_x):
    return _x if (_x % 2 == 0) else _x+1


def fit(img, maxsize=512):
    maxdim = max(*img.size)
    if maxdim > maxsize:
        # if True:
        ratio = maxsize/maxdim
        x, y = img.size
        size = (makeEven(int(x*ratio)), makeEven(int(y*ratio)))
        img = img.resize(size)
    return img


def warp(frame1, frame2, flo_path, blend=0.5, weights_path=None):
    flow21 = np.load(flo_path)
    frame1pil = np.array(frame1.convert('RGB').resize(
        (flow21.shape[1], flow21.shape[0])))
    frame1_warped21 = warp_flow(frame1pil, flow21)
    # frame2pil = frame1pil
    frame2pil = np.array(frame2.convert('RGB').resize(
        (flow21.shape[1], flow21.shape[0])))

    if weights_path:
        # TBD
        pass
    else:
        blended_w = frame2pil*(1-blend) + frame1_warped21*(blend)

    return PIL.Image.fromarray(blended_w.astype('uint8'))

# in_path = videoFramesFolder
# f'{in_path}/out_flo_fwd'

# in_path+'/temp_flo'
# in_path+'/out_flo_fwd'
# TBD flow backwards!

# os.chdir(models_dir)

# @title Generate optical flow and consistency maps
# @markdown Run once per init video


def generate_optical_flow(S):
    force_flow_generation = False  # @param {type:'boolean'}
    in_path = S.videoFramesFolder
    flo_folder = f'{in_path}/out_flo_fwd'

    if not S.video_init_flow_warp:
        print('video_init_flow_warp not set, skipping')

    if (S.animation_mode == 'Video Input') and (S.video_init_flow_warp):
        flows = glob(flo_folder+'/*.*')
        if (len(flows) > 0) and not force_flow_generation:
            print(
                f'Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {flo_folder}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again.')

        if (len(flows) == 0) or force_flow_generation:
            frames = sorted(glob(in_path+'/*.*'))
            if len(frames) < 2:
                print(
                    f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.')
            if len(frames) >= 2:

                raft_model = torch.nn.DataParallel(RAFT(args2))
                raft_model.load_state_dict(torch.load(
                    f'{S.root_path}/RAFT/models/raft-things.pth'))
                raft_model = raft_model.module.cuda().eval()

                for f in pathlib.Path(f'{S.flo_fwd_folder}').glob('*.*'):
                    f.unlink()

                temp_flo = in_path+'/temp_flo'
                flo_fwd_folder = in_path+'/out_flo_fwd'

                os.makedirs(flo_fwd_folder, exist_ok=True)
                os.makedirs(temp_flo, exist_ok=True)

                # TBD Call out to a consistency checker?

                for frame1, frame2 in tqdm(zip(frames[:-1], frames[1:]), total=len(frames)-1):

                    out_flow21_fn = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"

                    frame1 = load_img(frame1, S.width_height)
                    frame2 = load_img(frame2, S.width_height)

                    flow21 = get_flow(frame2, frame1, raft_model)
                    np.save(out_flow21_fn, flow21)

                    if S.video_init_check_consistency:
                        # TBD
                        pass

                del raft_model
                gc.collect()
