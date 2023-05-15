from urllib.parse import urlparse
import os
import hashlib
import lpips
import torchvision.transforms as T
import os
import cv2
import pandas as pd
import gc
import lpips
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
import numpy as np
import hashlib
from numpy import asarray

from .secondary_diffusion_model import SecondaryDiffusionImageNet2
from . import disco_utils

import comfy.model_management

diff_model_map = {
    '256x256_diffusion_uncond': { 'downloaded': False, 'sha': 'a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1', 'uri_list': ['https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', 'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'] },
    '512x512_diffusion_uncond_finetune_008100': { 'downloaded': False, 'sha': '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648', 'uri_list': ['https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt', 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'] },
    'portrait_generator_v001': { 'downloaded': False, 'sha': 'b7e8c747af880d4480b6707006f1ace000b058dd0eac5bb13558ba3752d9b5b9', 'uri_list': ['https://huggingface.co/felipe3dartist/portrait_generator_v001/resolve/main/portrait_generator_v001_ema_0.9999_1MM.pt'] },
    'pixelartdiffusion_expanded': { 'downloaded': False, 'sha': 'a73b40556634034bf43b5a716b531b46fb1ab890634d854f5bcbbef56838739a', 'uri_list': ['https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt'] },
    'pixel_art_diffusion_hard_256': { 'downloaded': False, 'sha': 'be4a9de943ec06eef32c65a1008c60ad017723a4d35dc13169c66bb322234161', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_hard_256/resolve/main/pixel_art_diffusion_hard_256.pt'] },
    'pixel_art_diffusion_soft_256': { 'downloaded': False, 'sha': 'd321590e46b679bf6def1f1914b47c89e762c76f19ab3e3392c8ca07c791039c', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_soft_256/resolve/main/pixel_art_diffusion_soft_256.pt'] },
    'pixelartdiffusion4k': { 'downloaded': False, 'sha': 'a1ba4f13f6dabb72b1064f15d8ae504d98d6192ad343572cc416deda7cccac30', 'uri_list': ['https://huggingface.co/KaliYuga/pixelartdiffusion4k/resolve/main/pixelartdiffusion4k.pt'] },
    'watercolordiffusion_2': { 'downloaded': False, 'sha': '49c281b6092c61c49b0f1f8da93af9b94be7e0c20c71e662e2aa26fee0e4b1a9', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion_2/resolve/main/watercolordiffusion_2.pt'] },
    'watercolordiffusion': { 'downloaded': False, 'sha': 'a3e6522f0c8f278f90788298d66383b11ac763dd5e0d62f8252c962c23950bd6', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion/resolve/main/watercolordiffusion.pt'] },
    'PulpSciFiDiffusion': { 'downloaded': False, 'sha': 'b79e62613b9f50b8a3173e5f61f0320c7dbb16efad42a92ec94d014f6e17337f', 'uri_list': ['https://huggingface.co/KaliYuga/PulpSciFiDiffusion/resolve/main/PulpSciFiDiffusion.pt'] },
    'secondary': { 'downloaded': False, 'sha': '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a', 'uri_list': ['https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth', 'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth', 'https://ipfs.pollinations.ai/ipfs/bafybeibaawhhk7fhyhvmm7x24zwwkeuocuizbqbcg5nqx64jq42j75rdiy/secondary_model_imagenet_2.pth'] },
}

class ModelSettings:
    def __init__(self, model_name, model_path):
        self.model_path = model_path
        #@markdown ####**Models Settings (note: For pixel art, the best is pixelartdiffusion_expanded):**
        self.diffusion_model = model_name #@param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]

        self.use_secondary_model = True #@param {type: 'boolean'}
        self.diffusion_sampling_mode = 'ddim' #@param ['plms','ddim']
        #@markdown #####**Custom model:**
        self.custom_path = '/content/drive/MyDrive/deep_learning/ddpm/ema_0.9999_058000.pt'#@param {type: 'string'}

        #@markdown #####**CLIP settings:**
        self.use_checkpoint = True #@param {type: 'boolean'}
        self.ViTB32 = True #@param{type:"boolean"}
        self.ViTB16 = True #@param{type:"boolean"}
        self.ViTL14 = False #@param{type:"boolean"}
        self.ViTL14_336px = False #@param{type:"boolean"}
        self.RN101 = False #@param{type:"boolean"}
        self.RN50 = True #@param{type:"boolean"}
        self.RN50x4 = False #@param{type:"boolean"}
        self.RN50x16 = False #@param{type:"boolean"}
        self.RN50x64 = False #@param{type:"boolean"}

        #@markdown #####**OpenCLIP settings:**
        self.ViTB32_laion2b_e16 = False #@param{type:"boolean"}
        self.ViTB32_laion400m_e31 = False #@param{type:"boolean"}
        self.ViTB32_laion400m_32 = False #@param{type:"boolean"}
        self.ViTB32quickgelu_laion400m_e31 = False #@param{type:"boolean"}
        self.ViTB32quickgelu_laion400m_e32 = False #@param{type:"boolean"}
        self.ViTB16_laion400m_e31 = False #@param{type:"boolean"}
        self.ViTB16_laion400m_e32 = False #@param{type:"boolean"}
        self.RN50_yffcc15m = False #@param{type:"boolean"}
        self.RN50_cc12m = False #@param{type:"boolean"}
        self.RN50_quickgelu_yfcc15m = False #@param{type:"boolean"}
        self.RN50_quickgelu_cc12m = False #@param{type:"boolean"}
        self.RN101_yfcc15m = False #@param{type:"boolean"}
        self.RN101_quickgelu_yfcc15m = False #@param{type:"boolean"}

        #@markdown If you're having issues with model downloads, check this to compare SHA's:
        self.check_model_SHA = False #@param{type:"boolean"}

        self.kaliyuga_pixel_art_model_names = ['pixelartdiffusion_expanded', 'pixel_art_diffusion_hard_256', 'pixel_art_diffusion_soft_256', 'pixelartdiffusion4k', 'PulpSciFiDiffusion']
        self.kaliyuga_watercolor_model_names = ['watercolordiffusion', 'watercolordiffusion_2']
        self.kaliyuga_pulpscifi_model_names = ['PulpSciFiDiffusion']
        self.diffusion_models_256x256_list = ['256x256_diffusion_uncond'] + self.kaliyuga_pixel_art_model_names + self.kaliyuga_watercolor_model_names + self.kaliyuga_pulpscifi_model_names


    def get_model_filename(self, diffusion_model_name):
        model_uri = diff_model_map[diffusion_model_name]['uri_list'][0]
        model_filename = os.path.basename(urlparse(model_uri).path)
        return model_filename

    def download_model(self, diffusion_model_name, uri_index=0):
        if diffusion_model_name != 'custom':
            model_filename = self.get_model_filename(diffusion_model_name)
            model_local_path = os.path.join(self.model_path, model_filename)
            if os.path.exists(model_local_path) and self.check_model_SHA:
                print(f'Checking {diffusion_model_name} File')
                with open(model_local_path, "rb") as f:
                    bytes = f.read()
                    hash = hashlib.sha256(bytes).hexdigest()
                if hash == diff_model_map[diffusion_model_name]['sha']:
                    print(f'{diffusion_model_name} SHA matches')
                    diff_model_map[diffusion_model_name]['downloaded'] = True
                else:
                    print(f"{diffusion_model_name} SHA doesn't match. Will redownload it.")
            elif os.path.exists(model_local_path) and not self.check_model_SHA or diff_model_map[diffusion_model_name]['downloaded']:
                print(f'{diffusion_model_name} already downloaded. If the file is corrupt, enable check_model_SHA.')
                diff_model_map[diffusion_model_name]['downloaded'] = True

            if not diff_model_map[diffusion_model_name]['downloaded']:
                for model_uri in diff_model_map[diffusion_model_name]['uri_list']:
                    disco_utils.wget(model_uri, self.model_path)
                    if os.path.exists(model_local_path):
                        diff_model_map[diffusion_model_name]['downloaded'] = True
                        return
                    else:
                        print(f'{diffusion_model_name} model download from {model_uri} failed. Will try any fallback uri.')
                print(f'{diffusion_model_name} download failed.')

    def setup(self, useCPU):
        # Download the diffusion model(s)
        self.download_model(self.diffusion_model)
        if self.use_secondary_model:
            self.download_model('secondary')

        self.model_config = model_and_diffusion_defaults()
        if self.diffusion_model == '512x512_diffusion_uncond_finetune_008100':
            self.model_config.update({
                'attention_resolutions': '32, 16, 8',
                'class_cond': False,
                'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
                'rescale_timesteps': True,
                'timestep_respacing': 250, #No need to edit this, it is taken care of later.
                'image_size': 512,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_checkpoint': self.use_checkpoint,
                'use_fp16': not useCPU,
                'use_scale_shift_norm': True,
            })
        elif self.diffusion_model == '256x256_diffusion_uncond':
            self.model_config.update({
                'attention_resolutions': '32, 16, 8',
                'class_cond': False,
                'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
                'rescale_timesteps': True,
                'timestep_respacing': 250, #No need to edit this, it is taken care of later.
                'image_size': 256,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_checkpoint': self.use_checkpoint,
                'use_fp16': not useCPU,
                'use_scale_shift_norm': True,
            })
        elif self.diffusion_model == 'portrait_generator_v001':
            self.model_config.update({
                'attention_resolutions': '32, 16, 8',
                'class_cond': False,
                'diffusion_steps': 1000,
                'rescale_timesteps': True,
                'image_size': 512,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 128,
                'num_heads': 4,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_checkpoint': self.use_checkpoint,
                'use_fp16': True,
                'use_scale_shift_norm': True,
            })
        else:  # E.g. A model finetuned by KaliYuga
            self.model_config.update({
                'attention_resolutions': '16',
                'class_cond': False,
                'diffusion_steps': 1000,
                'rescale_timesteps': True,
                'timestep_respacing': 'ddim100',
                'image_size': 256,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 128,
                'num_heads': 1,
                'num_res_blocks': 2,
                'use_checkpoint': self.use_checkpoint,
                'use_fp16': True,
                'use_scale_shift_norm': False,
            })

        self.model_default = self.model_config['image_size']

        device = comfy.model_management.get_torch_device()

        if self.use_secondary_model:
            self.secondary_model = SecondaryDiffusionImageNet2()
            self.secondary_model.load_state_dict(torch.load(f'{self.model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
            self.secondary_model.eval().requires_grad_(False).to(device)

        self.clip_models = []
        #if self.ViTB32: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device))
        #if self.ViTB16: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device))
        #if self.ViTL14: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device))
        #if self.ViTL14_336px: clip_models.append(clip.load('ViT-L/14@336px', jit=False)[0].eval().requires_grad_(False).to(device))
        #if self.RN50: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))
        #if self.RN50x4: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device))
        #if self.RN50x16: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device))
        #if self.RN50x64: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device))
        #if self.RN101: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device))
        #if self.ViTB32_laion2b_e16: clip_models.append(open_clip.create_model('ViT-B-32', pretrained='laion2b_e16').eval().requires_grad_(False).to(device))
        #if self.ViTB32_laion400m_e31: clip_models.append(open_clip.create_model('ViT-B-32', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
        #if self.ViTB32_laion400m_32: clip_models.append(open_clip.create_model('ViT-B-32', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
        #if self.ViTB32quickgelu_laion400m_e31: clip_models.append(open_clip.create_model('ViT-B-32-quickgelu', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
        #if self.ViTB32quickgelu_laion400m_e32: clip_models.append(open_clip.create_model('ViT-B-32-quickgelu', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
        #if self.ViTB16_laion400m_e31: clip_models.append(open_clip.create_model('ViT-B-16', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
        #if self.ViTB16_laion400m_e32: clip_models.append(open_clip.create_model('ViT-B-16', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
        #if self.RN50_yffcc15m: clip_models.append(open_clip.create_model('RN50', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
        #if self.RN50_cc12m: clip_models.append(open_clip.create_model('RN50', pretrained='cc12m').eval().requires_grad_(False).to(device))
        #if self.RN50_quickgelu_yfcc15m: clip_models.append(open_clip.create_model('RN50-quickgelu', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
        #if self.RN50_quickgelu_cc12m: clip_models.append(open_clip.create_model('RN50-quickgelu', pretrained='cc12m').eval().requires_grad_(False).to(device))
        #if self.RN101_yfcc15m: clip_models.append(open_clip.create_model('RN101', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
        #if self.RN101_quickgelu_yfcc15m: clip_models.append(open_clip.create_model('RN101-quickgelu', pretrained='yfcc15m').eval().requires_grad_(False).to(device))

        self.lpips_model = lpips.LPIPS(net='vgg').to(device)
