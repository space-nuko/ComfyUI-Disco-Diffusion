import os.path
from PIL import Image
import numpy as np
import comfy.model_management
from comfy.cli_args import args
import folder_paths as comfy_paths
import folder_paths
from pprint import pp

NODE_FILE = os.path.abspath(__file__)
DISCO_DIFFUSION_ROOT = os.path.dirname(NODE_FILE)

import sys
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "CLIP"))
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "MiDaS"))
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "ResizeRight"))
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "guided-diffusion"))
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "RAFT/core"))
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "open_clip/src"))


import torch
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from .settings import DiscoDiffusionSettings
from .model_settings import ModelSettings, diff_model_map
from .diffuse import diffuse
from .CLIP import clip as openai_clip
import open_clip


OPENAI_CLIP_MODELS = openai_clip.available_models()
OPEN_CLIP_MODELS = open_clip.list_pretrained()


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class OpenAICLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_map = {"required":{}}
        clip_models_base = OPENAI_CLIP_MODELS
        clip_models = OPEN_CLIP_MODELS
        
        # Valid models and their pretrains for Disco Diffusion
        valid_pretrained = ['N/A', 'laion2b_e16', 'laion400m_e31', 'laion400m_e32', 'yfcc15m', 'cc12m']
        valid_model = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 
                        'ViT-L/14@336px', 'RN50-quickgelu', 'RN101-quickgelu', 'ViT-B-32', 'ViT-B-32-quickgelu',
                        'ViT-B-16']

        # Create one model  dictionary
        for model in clip_models_base:
            clip_models.append((model, 'N/A')) # N/A so not to be excluded from openai check below
        clip_models = sorted(clip_models)
        
        # Create a valid list of CLIP Models in ComfyUI Input format.
        for model, pretrained in clip_models:
            if ( pretrained not in valid_pretrained
                    or model not in valid_model ): # Exclude broken openai and duplicates
                continue
            if model in ['ViT-B/32', 'ViT-B/16', 'RN50']: # Default DD CLIP Models
                options = (["True", "False"],)
            else:
                options = (["False", "True"],)
            input_map['required'].update({model: options})
            
        return input_map

    # These are technically different model formats so don't use them with vanilla nodes!
    RETURN_TYPES = ("GUIDED_CLIP",)
    FUNCTION = "load"

    CATEGORY = "loaders"

    def __init__(self):
        pass

    def load(self, **clip_model_names):
    
        clip_model_names = {key: True if value.lower() == 'true' else False for key, value in clip_model_names.items()}    
        clip_models = []
        
        device = comfy.model_management.get_torch_device()

        # For my own notes (because I was confused about this earlier):
        # DD requires the use of torch.autograd.grad so it can steer the output
        # image towards CLIP embeddings by calculating loss.
        # But it's more efficient to run operations on tensors loaded without
        # support for calculating loss, and most people aren't training models,
        # they're just running inference.
        # And "torch.autograd.grad" is a function mostly used for training.
        # But because (this implementation of) guided diffusion requires
        # autograd, we have to load the tensors with inference mode off
        # ourselves (ComfyUI enables it by default for maximum performance).
        # Same with the guided diffusion model, secondary diffusion model and
        # sampling code, it all must be loaded/run with support for autograd
        # (inference mode off).
        with torch.inference_mode(False):
            for model_name, activated in clip_model_names.items():
                if activated:
                    print(f'[Disco Diffusion] Loading CLIP model {model_name}')
                    if model_name in OPENAI_CLIP_MODELS:
                        clip_model = openai_clip.load(model_name, jit=False)[0]
                    else:
                        for model in OPEN_CLIP_MODELS:
                            if model_name not in model[0]:
                                continue
                            name, pretrained = model
                            clip_model = open_clip.create_model(name, pretrained=pretrained)
                            clip_model.eval().requires_grad_(False).to(device)
                    if clip_model not in clip_models:
                        clip_models.append(clip_model)

        return (clip_models, )


GUIDED_DIFFUSION_MODELS = list(diff_model_map.keys())


class GuidedDiffusionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (GUIDED_DIFFUSION_MODELS, { "default": "512x512_diffusion_uncond_finetune_008100" }),
            "use_checkpoint": (["True", "False"],),
            #"use_secondary": (["True", "False"],),
        }}

    # These are technically different model formats so don't use them with vanilla nodes!
    RETURN_TYPES = ("GUIDED_DIFFUSION_MODEL",)
    FUNCTION = "load"

    CATEGORY = "loaders"

    def __init__(self):
        pass

    def load(self, model_name, use_checkpoint, use_secondary=False):
        use_cpu = args.cpu
        
        use_checkpoint = True if use_checkpoint == "True" else False
        use_secondary = True if use_secondary == "True" else False
        
        with torch.inference_mode(False):
            model_settings = ModelSettings(model_name, os.path.join(folder_paths.models_dir, "Disco-Diffusion"), use_checkpoint, use_secondary='True')

        model_settings.setup(use_cpu)

        return (model_settings,)


class DiscoDiffusionExtraSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "eta": ("FLOAT", { "default": 0.8, "min": -1.0, "max": 1.0 }), # I couldn't find any setting breakdowns with a range beyond -1.0 to 1.0
            "cutn": ("INT", { "default": 16, "min": 1, "max": 32 }),
            "cutn_batches": ("INT", { "default": 2, "min": 1, "max": 16 }),
            "cut_overview": ("STRING", { "default": "[12]*400+[4]*600" }),
            "cut_innercut": ("STRING", { "default": "[4]*400+[12]*600" }),
            "cut_ic_pow": ("STRING", { "default": "[1]*1000" }),
            "cut_icgray_p": ("STRING", { "default": "[0.2]*400+[0]*600" }),
            "clamp_max": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0}),
            "clip_denoised": (["False","True"],),
            "perlin_init": (["False","True"],),
            "perlin_mode": (["mixed","color","gray"],),
            "use_horizontal_symmetry": (["False","True"],),
            "use_vertical_symmetry": (["False","True"],),
        }}

    # These are technically different model formats so don't use them with vanilla nodes!
    RETURN_TYPES = ("DISCO_DIFFUSION_EXTRA_SETTINGS",)
    FUNCTION = "make_settings"

    CATEGORY = "sampling"

    def __init__(self):
        pass

    def make_settings(self, eta, cutn, cutn_batches, cut_overview, cut_innercut, cut_ic_pow, cut_icgray_p, clamp_max, clip_denoised,
                        perlin_init, perlin_mode, use_horizontal_symmetry, use_vertical_symmetry):
        extra_settings = {
            "eta": eta,
            "cutn": cutn,
            "cutn_batches": cutn_batches,
            "cut_overview": cut_overview,
            "cut_innercut": cut_innercut,
            "cut_ic_pow": cut_ic_pow,
            "cut_icgray_p": cut_icgray_p,
            "clip_denoised": True if clip_denoised == 'True' else False,
            "perlin_init": True if perlin_init == 'True' else False,
            "perlin_mode": perlin_mode if perlin_mode in ['mixed', 'color', 'gray'] else 'mixed',
            "use_horizontal_symmetry": True if use_horizontal_symmetry == 'True' else False,
            "use_vertical_symmetry": True if use_vertical_symmetry == 'True' else False,     
        }

        return (extra_settings,)


DEFAULT_PROMPT = """\
; How to prompt:
; Each line is prefixed with the starting step number of the prompt.
; More than one line with the same step number concatenates the two prompts together.
; Each individual prompt can be no more than 77 CLIP tokens long.
; Weights are parsed from the end of each prompt with "25:a fluffy fox:5" syntax
; Comments are written with the ';' character. Blank lines are ignored.

0:A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade. Trending on artstation.
0:yellow color scheme
;100:This set of prompts starts at step 100.
;100:This prompt has weight five:5
""".strip()


class DiscoDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": DEFAULT_PROMPT, "multiline": True}),
                "guided_diffusion": ("GUIDED_DIFFUSION_MODEL",),
                "guided_clip": ("GUIDED_CLIP",),
                # Sane defaults:
                # 1280x768 for 512x512 models
                # 512x448 for 256x256 models
                "width": ("INT", {"default": 1280, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 250, "min": 1, "max": 10000}),
                "skip_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "n_batches": ("INT", {"default": 1, "min": 1, "max": 16}),
                # "max_frames": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "sampling_mode": (["plms", "ddim", "stsp", "ltsp"], {"default": "ddim"}),
                "clip_guidance_scale": ("FLOAT", { "default": 1500, "min": 1, "max": 10000000 }),
                "tv_scale": ("FLOAT", { "default": 0, "min": 0, "max": 100000 }),
                "range_scale": ("FLOAT", { "default": 150, "min": 0, "max": 100000 }),
                "sat_scale": ("FLOAT", { "default": 0, "min": 0, "max": 100000 }),
            },
            "optional": {
                "init_image": ("IMAGE",),
                "extra_settings": ("DISCO_DIFFUSION_EXTRA_SETTINGS",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"

    CATEGORY = "sampling"

    def __init__(self):
        pass

    def parse_prompts(self, text):
        result = {}
        for line in text.split('\n'):
            line = line.split(';')[0].strip()
            if line:
                if ':' in line:
                    vals = line.split(':', 2)
                    key = vals[0]
                    value = vals[1]
                    weight = "1"
                    if len(vals) >= 3:
                        weight = vals[2]

                    try:
                        key = int(key)
                    except ValueError:
                        weight = value
                        value = key
                        key = 0

                    if key in result:
                        result[key].append(value.strip() + ":" + weight)
                    else:
                        result[key] = [value.strip() + ":" + weight]
                else:
                    if 0 in result:
                        result[0].append(line)
                    else:
                        result[0] = [line]
        return result

    def load_model(self, model_settings, steps):
        device = comfy.model_management.get_torch_device()

        # Update Model Settings
        timestep_respacing = f'ddim{steps}'
        diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
        model_settings.model_config.update({
            'timestep_respacing': timestep_respacing,
            'diffusion_steps': diffusion_steps,
        })

        model, diffusion = create_model_and_diffusion(**model_settings.model_config)
        if model_settings.diffusion_model == 'custom':
            model.load_state_dict(torch.load(model_settings.custom_path, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(f'{model_settings.model_path}/{model_settings.get_model_filename(model_settings.diffusion_model)}', map_location='cpu'))
            model.requires_grad_(False).eval().to(device)

        for name, param in model.named_parameters():
            if 'qkv' in name or 'norm' in name or 'proj' in name:
                param.requires_grad_()

        if model_settings.model_config['use_fp16']:
            model.convert_to_fp16()

        return model, diffusion

    def generate(self, text, guided_diffusion, guided_clip, width, height, seed, steps, skip_steps, n_batches, sampling_mode,
                 clip_guidance_scale, tv_scale, range_scale, sat_scale, extra_settings=None, init_image=None):
        clip_vision = guided_clip # This should be further removed down to the do_run.py
        settings = DiscoDiffusionSettings()
        settings.seed = seed
        settings.steps = steps
        settings.skip_steps = skip_steps
        settings.n_batches = n_batches
        settings.max_frames = 1
        settings.text_prompts = self.parse_prompts(text)
        settings.width_height_for_256x256_models = [width, height]
        settings.width_height_for_512x512_models = [width, height]
        settings.clip_guidance_scale = clip_guidance_scale
        settings.tv_scale = tv_scale
        settings.range_scale = range_scale
        settings.sat_scale = sat_scale
        if init_image != None:
            tmp_path = os.path.join(comfy_paths.temp_directory, 'dd_init_image_temp.png')
            os.makedirs(comfy_paths.temp_directory, exist_ok=True)
            tensor2pil(init_image).save(tmp_path)
            if hasattr(settings, 'init_image'):
                settings.init_image = tmp_path
            else:
                setattr(settings, 'init_image', tmp_path)
            
        guided_diffusion.diffusion_sampling_mode = sampling_mode
            
        # Set extra settings
        if extra_settings is not None:
            for extra_name, extra_value in extra_settings.items():
                if hasattr(settings, extra_name):
                    setattr(settings, extra_name, extra_value)
                else:
                    print(f"[Disco Diffusion] The requested extra setting `{extra_name}` is not valid.")

        print("[Disco Diffusion] Parsed Prompts:")
        pp(settings.text_prompts)

        settings.setup(guided_diffusion)

        # Have to defer loading the model until here since step count isn't
        # known until now
        print(f"[Disco Diffusion]: Loading diffusion model {guided_diffusion.diffusion_model}")
        with torch.inference_mode(False):
            model, diffusion = self.load_model(guided_diffusion, settings.steps)
        
        print("[Disco Diffusion]: Starting diffusion...")
        with torch.inference_mode(False):
            images = diffuse(model, diffusion, guided_clip, clip_vision, settings, 0)

        return (images,)


NODE_CLASS_MAPPINGS = {
    "DiscoDiffusion_OpenAICLIPLoader": OpenAICLIPLoader,
    "DiscoDiffusion_GuidedDiffusionLoader": GuidedDiffusionLoader,
    "DiscoDiffusion_DiscoDiffusion": DiscoDiffusion,
    "DiscoDiffusion_DiscoDiffusionExtraSettings": DiscoDiffusionExtraSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiscoDiffusion_OpenAICLIPLoader": "Guided Diffusion CLIP Loader",
    "DiscoDiffusion_GuidedDiffusionLoader": "Guided Diffusion Loader",
    "DiscoDiffusion_DiscoDiffusion": "Disco Diffusion Sampler",
    "DiscoDiffusion_DiscoDiffusionExtraSettings": "Disco Diffusion Extra Settings",
}
