import os.path
import comfy.model_management

NODE_FILE = os.path.abspath(__file__)
DISCO_DIFFUSION_ROOT = os.path.dirname(NODE_FILE)

import sys
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "CLIP"))
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "MiDaS"))
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "ResizeRight"))
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "guided-diffusion"))
sys.path.append(os.path.join(DISCO_DIFFUSION_ROOT, "RAFT/core"))


from .CLIP import clip
from .settings import DiscoDiffusionSettings
from .model_settings import ModelSettings
from .diffuse import diffuse


class DiscoDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ()
    FUNCTION = "generate"

    CATEGORY = "sampling"

    OUTPUT_NODE = True

    def __init__(self):
        self.settings = DiscoDiffusionSettings()
        self.model_settings = ModelSettings()
        self.settings.setup(self.model_settings)
        self.model_settings.setup(self.settings)

    def generate(self, text, seed):
        device = comfy.model_management.get_torch_device()
        clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
        diffuse(clip_model, clip_model, self.settings, 0)
        return { "ui": { "images": {} } }


NODE_CLASS_MAPPINGS = {
    "ComfyUI_DiscoDiffusion": DiscoDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_DiscoDiffusion": "Disco Diffusion",
}
