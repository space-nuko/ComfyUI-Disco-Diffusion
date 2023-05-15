import os.path
import comfy.model_management
from comfy.cli_args import args
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


OPENAI_CLIP_MODELS = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101']
OPEN_CLIP_MODELS = [
    ['ViT-B-32', 'laion2b_e16'],
    ['ViT-B-32', 'laion400m_e31'],
    ['ViT-B-32', 'laion400m_e32'],
    ['ViT-B-32-quickgelu', 'laion400m_e31'],
    ['ViT-B-32-quickgelu', 'laion400m_e32'],
    ['ViT-B-16', 'laion400m_e31'],
    ['ViT-B-16', 'laion400m_e32'],
    ['RN50', 'yfcc15m'],
    ['RN50', 'cc12m'],
    ['RN50-quickgelu', 'yfcc15m'],
    ['RN50-quickgelu', 'cc12m'],
    ['RN101', 'yfcc15m'],
    ['RN101-quickgelu', 'yfcc15m']
]


class OpenAICLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        open_clip_models = ["_".join(m) for m in OPEN_CLIP_MODELS]
        return {"required": {"model_name": (OPENAI_CLIP_MODELS + open_clip_models, { "default": "ViT-B/32" }) }}

    # These are technically different model formats so don't use them with vanilla nodes!
    RETURN_TYPES = ("CLIP", "CLIP_VISION")
    FUNCTION = "load"

    CATEGORY = "loaders"

    def __init__(self):
        pass

    def load(self, model_name):
        device = comfy.model_management.get_torch_device()

        if model_name in OPENAI_CLIP_MODELS:
            clip_model = openai_clip.load(model_name, jit=False)[0]
        else:
            spl = model_name.split("_")
            clip_model = open_clip.create_model(spl[0], pretrained=spl[1])

        clip_model.eval().requires_grad_(False).to(device)

        return (clip_model, clip_model,)


GUIDED_DIFFUSION_MODELS = list(diff_model_map.keys())


class GuidedDiffusionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (GUIDED_DIFFUSION_MODELS, { "default": "512x512_diffusion_uncond_finetune_008100" }) }}

    # These are technically different model formats so don't use them with vanilla nodes!
    RETURN_TYPES = ("GUIDED_DIFFUSION_MODEL",)
    FUNCTION = "load"

    CATEGORY = "loaders"

    def __init__(self):
        pass

    def load(self, model_name):
        use_cpu = args.cpu
        model_settings = ModelSettings(model_name, os.path.join(folder_paths.models_dir, "Disco-Diffusion"))

        model_settings.setup(use_cpu)

        return (model_settings,)


DEFAULT_PROMPT = """\
# How to prompt:
# Each line is prefixed with the starting frame number of the prompt.
# More than one line with the same frame number concatenates the two prompts together.
# Each individual prompt can be no more than 77 characters long.
# Weights are parsed from the end of each prompt with "25:a fluffy fox:5" syntax
# Comments are written with the '#' character. Blank lines are ignored.

0:A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade. Trending on artstation.
0:yellow color scheme
#100:This set of prompts start at frame 100.
#100:This prompt has weight five:5
""".strip()


class DiscoDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"default": DEFAULT_PROMPT, "multiline": True}),
                             "guided_diffusion": ("GUIDED_DIFFUSION_MODEL",),
                             "clip": ("CLIP",),
                             "clip_vision": ("CLIP_VISION",),
                             # Sane defaults:
                             # 1280x768 for 512x512 models
                             # 512x448 for 256x256 models
                             "width": ("INT", {"default": 1280, "min": 64, "max": 2048, "step": 64}),
                             "height": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 64}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "steps": ("INT", {"default": 250, "min": 1, "max": 10000}),
                             "skip_steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                             "n_batches": ("INT", {"default": 1, "min": 1, "max": 16}),
                             # "max_frames": ("INT", {"default": 1, "min": 1, "max": 1000}),
                             "sampling_mode": (["plms", "ddim", "stsp", "ltsp"], {"default": "ddim"}),
                             "clip_guidance_scale": ("INT", { "default": 5000, "min": 1, "max": 10000000 }),
                             "tv_scale": ("INT", { "default": 0, "min": 0, "max": 100000 }),
                             "range_scale": ("INT", { "default": 150, "min": 0, "max": 100000 }),
                             "sat_scale": ("INT", { "default": 0, "min": 0, "max": 100000 }),
                             "eta": ("FLOAT", { "default": 0.8, "min": 0, "max": 100 }),
                             "cutn": ("INT", { "default": 16, "min": 1, "max": 32 }),
                             "cutn_batches": ("INT", { "default": 2, "min": 1, "max": 16 }),
                             "cut_overview": ("STRING", { "default": "[12]*400+[4]*600" }),
                             "cut_innercut": ("STRING", { "default": "[4]*400+[12]*600" }),
                             "cut_ic_pow": ("STRING", { "default": "[1]*1000" }),
                             "cut_icgray_p": ("STRING", { "default": "[0.2]*400+[0]*600" }),
                             }}
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"

    CATEGORY = "sampling"

    def __init__(self):
        pass

    def parse_prompts(self, text):
        result = {}
        for line in text.split('\n'):
            line = line.split('#')[0].strip()
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

    def generate(self, text, guided_diffusion, clip, clip_vision, width, height, seed, steps, skip_steps, n_batches, sampling_mode,
                 clip_guidance_scale, tv_scale, range_scale, sat_scale, eta,
                 cutn, cutn_batches, cut_overview, cut_innercut, cut_ic_pow, cut_icgray_p):
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
        settings.eta = eta
        settings.cutn = cutn
        settings.cutn_batches = cutn_batches
        settings.cut_overview = cut_overview
        settings.cut_innercut = cut_innercut
        settings.cut_ic_pow = cut_ic_pow
        settings.cut_icgray_p = cut_icgray_p
        guided_diffusion.diffusion_sampling_mode = sampling_mode

        print("[Disco Diffusion] Parsed Prompts:")
        pp(settings.text_prompts)

        settings.setup(guided_diffusion)

        # Have to defer loading the model until here since step count isn't
        # known until now
        model, diffusion = self.load_model(guided_diffusion, settings.steps)

        images = diffuse(model, diffusion, clip, clip_vision, settings, 0)

        return (images,)


NODE_CLASS_MAPPINGS = {
    "ComfyUI_OpenAICLIPLoader": OpenAICLIPLoader,
    "ComfyUI_GuidedDiffusionLoader": GuidedDiffusionLoader,
    "ComfyUI_DiscoDiffusion": DiscoDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_OpenAICLIPLoader": "OpenAI CLIP Loader",
    "ComfyUI_GuidedDiffusionLoader": "Guided Diffusion Loader",
    "ComfyUI_DiscoDiffusion": "Disco Diffusion",
}
