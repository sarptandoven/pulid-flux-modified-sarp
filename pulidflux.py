import types
import zipfile

import cv2
import torch
from insightface.utils.download import download_file
from insightface.utils.storage import BASE_REPO_URL
from insightface.utils import face_align
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional
import os
import logging
import folder_paths
import comfy
from insightface.app import FaceAnalysis
from .face_restoration_helper import FaceRestoreHelper, get_face_by_index, draw_on

from comfy import model_management
from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .encoders_flux import IDFormer, PerceiverAttentionCA

from .PulidFluxHook import pulid_forward_orig, set_model_dit_patch_replace, pulid_enter, pulid_patch_double_blocks_after
from .patch_util import PatchKeys, add_model_patch_option, set_model_patch

def set_extra_config_model_path(extra_config_models_dir_key, models_dir_name:str):
    models_dir_default = os.path.join(folder_paths.models_dir, models_dir_name)
    if extra_config_models_dir_key not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[extra_config_models_dir_key] = (
            [os.path.join(folder_paths.models_dir, models_dir_name)], folder_paths.supported_pt_extensions)
    else:
        if not os.path.exists(models_dir_default):
            os.makedirs(models_dir_default, exist_ok=True)
        folder_paths.add_model_folder_path(extra_config_models_dir_key, models_dir_default, is_default=True)

set_extra_config_model_path("pulid", "pulid")
set_extra_config_model_path("insightface", "insightface")
set_extra_config_model_path("facexlib", "facexlib")

INSIGHTFACE_DIR = folder_paths.get_folder_paths("insightface")[0]
FACEXLIB_DIR = folder_paths.get_folder_paths("facexlib")[0]

class PulidFluxModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.double_interval = 2
        self.single_interval = 4

        # Init encoder
        self.pulid_encoder = IDFormer()

        # Init attention
        num_ca = 19 // self.double_interval + 38 // self.single_interval
        if 19 % self.double_interval != 0:
            num_ca += 1
        if 38 % self.single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA() for _ in range(num_ca)
        ])

    def from_pretrained(self, path: str):
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

        del state_dict
        del state_dict_dict

    def get_embeds(self, face_embed, clip_embeds):
        return self.pulid_encoder(face_embed, clip_embeds)

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor

def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

wrappers_name = "PULID_wrappers"

class PulidFluxModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pulid_file": (folder_paths.get_filename_list("pulid"), )}}

    RETURN_TYPES = ("PULIDFLUX",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    def load_model(self, pulid_file):
        model_path = folder_paths.get_full_path("pulid", pulid_file)

        # Also initialize the model, takes longer to load but then it doesn't have to be done every time you change parameters in the apply node
        offload_device = model_management.unet_offload_device()
        load_device = model_management.get_torch_device()

        model = PulidFluxModel()

        logging.info("Loading PuLID-Flux model.")
        model.from_pretrained(path=model_path)

        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        del model

        return (model_patcher,)

def download_insightface_model(sub_dir, name, force=False, root='~/.insightface'):
    # Copied and modified from insightface.utils.storage.download
    # Solve https://github.com/deepinsight/insightface/issues/2711
    _root = os.path.expanduser(root)
    dir_path = os.path.join(_root, sub_dir, name)
    if os.path.exists(dir_path) and not force:
        return dir_path
    print('download_path:', dir_path)
    zip_file_path = os.path.join(_root, sub_dir, name + '.zip')
    model_url = "%s/%s.zip"%(BASE_REPO_URL, name)
    download_file(model_url,
             path=zip_file_path,
             overwrite=True)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # zip file has contains ${name}
    real_dir_path = os.path.join(_root, sub_dir)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(real_dir_path)
    #os.remove(zip_file_path)
    return dir_path

class PulidFluxInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insightface"
    CATEGORY = "pulid"

    def load_insightface(self, provider):
        name = "antelopev2"
        download_insightface_model("models", name, root=INSIGHTFACE_DIR)
        model = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider', ]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)

class PulidFluxEvaClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load_eva_clip"
    CATEGORY = "pulid"

    def load_eva_clip(self):
        from .eva_clip.factory import create_model_and_transforms

        clip_file_path = folder_paths.get_full_path("text_encoders", 'EVA02_CLIP_L_336_psz14_s6B.pt')
        if clip_file_path is None:
            clip_dir = os.path.join(folder_paths.models_dir, "clip")
        else:
            clip_dir = os.path.dirname(clip_file_path)
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True, local_dir=clip_dir)

        model = model.visual

        eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            model["image_mean"] = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            model["image_std"] = (eva_transform_std,) * 3

        return (model,)

class ApplyPulidFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid_flux": ("PULIDFLUX", ),
                "eva_clip": ("EVA_CLIP", ),
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            },
            "optional": {
                "attn_mask": ("MASK", ),
                "options": ("OPTIONS",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_pulid_flux"
    CATEGORY = "pulid"

    def apply_pulid_flux(self, model, pulid_flux, eva_clip, face_analysis, image, weight, start_at, end_at, attn_mask=None, options={}, unique_id=None):
        model = model.clone()

        device = comfy.model_management.get_torch_device()
        dtype = model.model.diffusion_model.dtype

        # Process image
        image = image.to(device=device, dtype=dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=device, dtype=dtype)

        # Get face embeddings
        face_helper = FaceRestoreHelper(device=device)
        face_helper.clean_all()
        face_helper.read_image(image)
        face_helper.get_face_landmarks_5(only_center_face=False)
        face_helper.align_warp_face()

        if len(face_helper.cropped_faces) == 0:
            logging.warning("No face detected in the image.")
            return (model,)

        # Process faces
        face_embeds = []
        for face in face_helper.cropped_faces:
            face_tensor = image_to_tensor(face)
            face_tensor = face_tensor.unsqueeze(0).to(device=device, dtype=dtype)
            face_embed = face_analysis.get(face_tensor)
            if face_embed is not None:
                face_embeds.append(face_embed)

        if not face_embeds:
            logging.warning("No valid face embeddings generated.")
            return (model,)

        # Get CLIP embeddings
        clip_embeds = []
        for face in face_helper.cropped_faces:
            face_tensor = image_to_tensor(face)
            face_tensor = face_tensor.unsqueeze(0).to(device=device, dtype=dtype)
            clip_embed = eva_clip(face_tensor)
            clip_embeds.append(clip_embed)

        # Get PuLID embeddings
        pulid_embeds = []
        for face_embed, clip_embed in zip(face_embeds, clip_embeds):
            pulid_embed = pulid_flux.get_embeds(face_embed, clip_embed)
            pulid_embeds.append(pulid_embed)

        # Set up model patches
        model_patches = {}
        model_patches[wrappers_name] = {
            "pulid_data": {
                str(i): {
                    "embedding": embed,
                    "weight": weight,
                    "sigma_start": start_at,
                    "sigma_end": end_at
                } for i, embed in enumerate(pulid_embeds)
            }
        }

        # Apply patches
        set_model_patch(model, model_patches)

        return (model,)

class FixPulidFluxPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "fix_pulid_patch"
    CATEGORY = "pulid"

    def fix_pulid_patch(self, model):
        model = model.clone()
        model_patches = model.model_options.get(wrappers_name, {})
        if "pulid_data" in model_patches:
            del model_patches["pulid_data"]
        return (model,)

class PulidFluxOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_faces_order": (
                    ["left-right","right-left","top-bottom","bottom-top","small-large","large-small"],
                    {
                        "default": "large-small",
                        "tooltip": "left-right: Sort the left boundary of bbox by column from left to right.\n"
                                   "right-left: Reverse order of left-right (Sort the left boundary of bbox by column from right to left).\n"
                                   "top-bottom: Sort the top boundary of bbox by row from top to bottom.\n"
                                   "bottom-top: Reverse order of top-bottom (Sort the top boundary of bbox by row from bottom to top).\n"
                                   "small-large: Sort the area of bbox from small to large.\n"
                                   "large-small: Sort the area of bbox from large to small."
                    }
                ),
                "input_faces_index": ("INT",
                                      {
                                          "default": 0, "min": 0, "max": 1000, "step": 1,
                                          "tooltip": "If the value is greater than the size of bboxes, will set value to 0."
                                      }),
                "input_faces_align_mode": ("INT",
                                      {
                                          "default": 1, "min": 0, "max": 1, "step": 1,
                                          "tooltip": "Align face mode.\n"
                                                     "0: align_face and embed_face use different detectors. The results maybe different.\n"
                                                     "1: align_face and embed_face use the same detector."
                                      }),
            }
        }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "execute"
    CATEGORY = "pulid"

    def execute(self, input_faces_order, input_faces_index, input_faces_align_mode=1):
        return ({
            "input_faces_order": input_faces_order,
            "input_faces_index": input_faces_index,
            "input_faces_align_mode": input_faces_align_mode
        },)

class PulidFluxFaceDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE",),
                "options": ("OPTIONS",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("embed_face", "align_face", "face_bbox_image",)
    FUNCTION = "execute"
    CATEGORY = "pulid"
    OUTPUT_IS_LIST = (True, True, True,)

    def execute(self, face_analysis, image, options):
        device = comfy.model_management.get_torch_device()
        face_helper = FaceRestoreHelper(device=device)
        face_helper.clean_all()
        face_helper.read_image(image)
        face_helper.get_face_landmarks_5(only_center_face=False)
        face_helper.align_warp_face()

        if len(face_helper.cropped_faces) == 0:
            return ([], [], [])

        # Sort faces based on options
        bboxes = face_helper.bboxes
        if options["input_faces_order"] == "left-right":
            bboxes = sorted(bboxes, key=lambda x: x[0])
        elif options["input_faces_order"] == "right-left":
            bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
        elif options["input_faces_order"] == "top-bottom":
            bboxes = sorted(bboxes, key=lambda x: x[1])
        elif options["input_faces_order"] == "bottom-top":
            bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
        elif options["input_faces_order"] == "small-large":
            bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        elif options["input_faces_order"] == "large-small":
            bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

        # Get face index
        face_index = min(options["input_faces_index"], len(bboxes) - 1)
        if face_index < 0:
            face_index = 0

        # Get faces
        embed_faces = []
        align_faces = []
        face_bbox_images = []

        for i, bbox in enumerate(bboxes):
            if i == face_index:
                face = face_helper.cropped_faces[i]
                embed_faces.append(image_to_tensor(face))
                align_faces.append(image_to_tensor(face))
                face_bbox_images.append(draw_on(image, bbox))

        return (embed_faces, align_faces, face_bbox_images)

def crop_image(image, bbox, margin=0):
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return image[y1:y2, x1:x2]

def set_hook(diffusion_model, target_forward_orig):
    if not hasattr(diffusion_model, 'old_forward_orig_for_pulid'):
        diffusion_model.old_forward_orig_for_pulid = diffusion_model.forward_orig
    diffusion_model.forward_orig = target_forward_orig

def clean_hook(diffusion_model):
    if hasattr(diffusion_model, 'old_forward_orig_for_pulid'):
        diffusion_model.forward_orig = diffusion_model.old_forward_orig_for_pulid
        del diffusion_model.old_forward_orig_for_pulid

def pulid_outer_sample_wrappers_with_override(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    return wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)

def pulid_outer_sample_wrappers(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    return wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)

def pulid_apply_model_wrappers(wrapper_executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
    return wrapper_executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

NODE_CLASS_MAPPINGS = {
    "PulidFluxModelLoader": PulidFluxModelLoader,
    "PulidFluxInsightFaceLoader": PulidFluxInsightFaceLoader,
    "PulidFluxEvaClipLoader": PulidFluxEvaClipLoader,
    "ApplyPulidFlux": ApplyPulidFlux,
    "FixPulidFluxPatch": FixPulidFluxPatch,
    "PulidFluxOptions": PulidFluxOptions,
    "PulidFluxFaceDetector": PulidFluxFaceDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulidFluxModelLoader": "Load PuLID Flux Model",
    "PulidFluxInsightFaceLoader": "Load InsightFace (PuLID Flux)",
    "PulidFluxEvaClipLoader": "Load Eva Clip (PuLID Flux)",
    "ApplyPulidFlux": "Apply PuLID Flux",
    "FixPulidFluxPatch": "Fix PuLID Flux Patch",
    "PulidFluxOptions": "Pulid Flux Options",
    "PulidFluxFaceDetector": "Pulid Flux Face Detector",
}
