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


from .settings import DiscoDiffusionSettings
from .model_settings import ModelSettings
from .diffuse import diffuse


# class DiscoDiffusionCLIPLoader:
#     """
#     Loader for CLIP models compatible with Disco Diffusion (VIT-B)
#     """

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "clip_model_name": (folder_paths.get_filename_list("style_models"), )}}

#     RETURN_TYPES = ()
#     FUNCTION = "generate"

#     CATEGORY = "sampling"

#     OUTPUT_NODE = True

#     def __init__(self):
#         self.settings = DiscoDiffusionSettings()
#         self.model_settings = ModelSettings()
#         self.settings.setup(self.model_settings)
#         self.model_settings.setup(self.settings)

#     def generate(self, clip, clip_vision, text, seed):
#         device = comfy.model_management.get_torch_device()
#         diffuse(clip, clip_vision, self.settings, 0)
#         return { "ui": { "images": {} } }


class DiscoDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "clip": ("CLIP",),
                             "clip_vision": ("CLIP_VISION",),
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

    def generate(self, clip, clip_vision, text, seed):
        device = comfy.model_management.get_torch_device()
        diffuse(clip, clip_vision, self.settings, 0)
        return { "ui": { "images": {} } }


NODE_CLASS_MAPPINGS = {
    "ComfyUI_DiscoDiffusion": DiscoDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_DiscoDiffusion": "Disco Diffusion",
}
