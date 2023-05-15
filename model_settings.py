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

    'concept_art_generator_v000-1_alpha': { 'downloaded': False, 'sha': '0c0394148b4fb56234baee533c36f17202b8f936f28427a957f342aa2f18d040', 'uri_list': ['https://huggingface.co/WAS/concept_art_generator/resolve/main/concept_art_generator_v000-1_alpha.pt'] },
    'concept_art_generator_v000-2_alpha': { 'downloaded': False, 'sha': 'a7aee6b8dd73b6f3d377a8c579b99ed2922fc472ead16238074e08bf7517d075', 'uri_list': ['https://huggingface.co/WAS/concept_art_generator/resolve/main/concept_art_generator_v000-2_alpha.pt'] },

    'portrait_generator_v001': { 'downloaded': False, 'sha': 'b7e8c747af880d4480b6707006f1ace000b058dd0eac5bb13558ba3752d9b5b9', 'uri_list': ['https://huggingface.co/felipe3dartist/portrait_generator_v001/resolve/main/portrait_generator_v001_ema_0.9999_1MM.pt'] },
    'portrait_generator_v002': { 'downloaded': False, 'sha': '3bc39e28fd9690dafbbee83cc08d089a3640eca8ed4d14280c4a9d342c56fd7f', 'uri_list': ['https://huggingface.co/WAS/portrait-diffusion/resolve/main/ema_0.9999_165000.pt'] },
    'portrait_generator_v003': { 'downloaded': False, 'sha': 'c0e1739731efe682f6429f2f7f905104b602b415ef8bf0b507034cce6050e8e3', 'uri_list': ['https://huggingface.co/WAS/portrait-diffusion/resolve/main/ema_0.9999_430000.pt'] },
    'portrait_generator_v004': { 'downloaded': False, 'sha': '09daed35b70670b8491f27ef172e2a62f08f3a78326aeb88d23cade1fbb878ab', 'uri_list': ['https://huggingface.co/WAS/portrait-diffusion/resolve/main/ema_0.9999_080000.pt'] },
    'portrait_generator_v005': { 'downloaded': False, 'sha': '47a9e2cb9ddc9ca9adf9a6bde90c5107c74828b02fe7300bcf0a7d0fb30d5abc', 'uri_list': ['https://huggingface.co/felipe3dartist/Portrait_generator_V2.0/resolve/main/portrait_generator_v2.pt'] },

    'Architecture_Diffusion_1-5m': { 'downloaded': False, 'sha': '9a82579a5490e06fde5ff8ac9a37082f66ce10bb03f434b047596f6526ffd95e', 'uri_list': ['https://huggingface.co/jerostephan/Architecture_Diffusion_1.5M/resolve/main/Architecture_Diffusion_1.5M.pt'] },

    'pixelartdiffusion_expanded': { 'downloaded': False, 'sha': 'a73b40556634034bf43b5a716b531b46fb1ab890634d854f5bcbbef56838739a', 'uri_list': ['https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt'] },
    'pixel_art_diffusion_hard_256': { 'downloaded': False, 'sha': 'be4a9de943ec06eef32c65a1008c60ad017723a4d35dc13169c66bb322234161', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_hard_256/resolve/main/pixel_art_diffusion_hard_256.pt'] },
    'pixel_art_diffusion_soft_256': { 'downloaded': False, 'sha': 'd321590e46b679bf6def1f1914b47c89e762c76f19ab3e3392c8ca07c791039c', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_soft_256/resolve/main/pixel_art_diffusion_soft_256.pt'] },
    'pixelartdiffusion4k': { 'downloaded': False, 'sha': 'a1ba4f13f6dabb72b1064f15d8ae504d98d6192ad343572cc416deda7cccac30', 'uri_list': ['https://huggingface.co/KaliYuga/pixelartdiffusion4k/resolve/main/pixelartdiffusion4k.pt'] },
    'watercolordiffusion_2': { 'downloaded': False, 'sha': '49c281b6092c61c49b0f1f8da93af9b94be7e0c20c71e662e2aa26fee0e4b1a9', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion_2/resolve/main/watercolordiffusion_2.pt'] },
    'watercolordiffusion': { 'downloaded': False, 'sha': 'a3e6522f0c8f278f90788298d66383b11ac763dd5e0d62f8252c962c23950bd6', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion/resolve/main/watercolordiffusion.pt'] },
    'PulpSciFiDiffusion': { 'downloaded': False, 'sha': 'b79e62613b9f50b8a3173e5f61f0320c7dbb16efad42a92ec94d014f6e17337f', 'uri_list': ['https://huggingface.co/KaliYuga/PulpSciFiDiffusion/resolve/main/PulpSciFiDiffusion.pt'] },

    'Liminal_Diffusion_v1': { 'downloaded': False, 'sha': '87c36b544a367fceb0ca127d0028cd8a6f6b6e069e529b22999259d69c14f042', 'uri_list': ['https://huggingface.co/BrainArtLabs/liminal_diffusion/resolve/main/liminal_diffusion_v1.pt'] },
    'Liminal_Diffusion_Source': { 'downloaded': False, 'sha': 'ce0064b8cea56c8adb4e5aa0ee2d02f65cd8f1baa905cc6504f462f1aac6d6f4', 'uri_list': ['https://huggingface.co/BrainArtLabs/liminal_diffusion/resolve/main/liminal_diffusion_source.pt'] },
    'Medieval_Diffusion': { 'downloaded': False, 'sha': '1b66a0c9749f88b2d9124af7a3ba5c12d2645ffcc33356326269ccda8643a01b', 'uri_list': ['https://huggingface.co/KaliYuga/medievaldiffusion/resolve/main/medievaldiffusion.pt'] },
    'Lithography_Diffusion': { 'downloaded': False, 'sha': 'a3e6522f0c8f278f90788298d66383b11ac763dd5e0d62f8252c962c23950bd6', 'uri_list': ['https://huggingface.co/KaliYuga/lithographydiffusion/resolve/main/lithographydiffusion.pt'] },
    'Floral_Diffusion': { 'downloaded': False, 'sha': '197e9068f1ca0248fd89f9d6ca1f6f851783e04fddd0cc5b0020bfa655ed3aeb', 'uri_list': ['https://huggingface.co/jags/floraldiffusion/resolve/main/floraldiffusion.pt', 'https://www.dropbox.com/s/i0xrhq28ls1e94g/floraldiffusion.pt'] },
    'FeiArt_Handpainted_CG_Diffusion': { 'downloaded': False, 'sha': '85f95f0618f288476ffcec9f48160542ba626f655b3df963543388dcd059f86a', 'uri_list': ['https://huggingface.co/Feiart/FeiArt-Handpainted-CG-Diffusion/resolve/main/FeiArt-Handpainted-CG-Diffusion.pt'] },
    'Textile_Diffusion': { 'downloaded': False, 'sha': '82aa9ac10c67a806929b5399f04b933ffaa98f1a2bb0c18103d6ccd8f5bd2dd4', 'uri_list': ['https://huggingface.co/KaliYuga/textilediffusion/resolve/main/textilediffusion.pt'] },
    'Isometric_Diffusion_Revrart512px': { 'downloaded': False, 'sha': '649bb7d10ea5170b71bc24adfb17f0305955fde38d2bb1bc5428c6d5baf9811c', 'uri_list': ['https://huggingface.co/Revrart/IsometricDiffusionRevrart512px/resolve/main/IsometricDiffusionRevrart512px.pt'] },
    'Laproper_Diffusion_Deepspace_256': { 'downloaded': False, 'sha': '7c0d742d714ee512edda3299b440f22c72e95fda3f32e24c4ba1f03e0c4b0524', 'uri_list': ['https://huggingface.co/laproper/diffusion-deepspace-256/resolve/main/ema_0.9999_102000.pt']},
    'Schnippi_Diffusion_512x512_V2': { 'downloaded': False, 'sha': '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648', 'uri_list': ['https://huggingface.co/shnippi/shnippi_diffusion/resolve/main/512x512_shnippi_V2.pt']},

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
                    print(f'{diffusion_model_name} model downloading...')
                    disco_utils.pyget(model_uri, self.model_path, progress=True)
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
