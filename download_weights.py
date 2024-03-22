import os
import shutil

MODEL_CACHE = "diffusers-cache"
BASE_MODEL_PATH = "./weights"
CONTROLNET_MODEL_CANNY = "lllyasviel/control_v11p_sd15_canny"
CHECKPOINT_CYBERREALISTICv42 = "philz1337x/cyberrealistic-v4.2"

if os.path.exists(MODEL_CACHE):
    shutil.rmtree(MODEL_CACHE)
os.makedirs(MODEL_CACHE)

import torch
from diffusers import ControlNetModel, StableDiffusionPipeline

TMP_CACHE = "tmp_cache"

if os.path.exists(TMP_CACHE):
    shutil.rmtree(TMP_CACHE)
os.makedirs(TMP_CACHE)

cn = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL_CANNY,
    torch_dtype=torch.float16,
    cache_dir=TMP_CACHE,
)
cn.half()
cn.save_pretrained(os.path.join(MODEL_CACHE, 'canny'))

pipe = StableDiffusionPipeline.from_pretrained(
    CHECKPOINT_CYBERREALISTICv42,
    torch_dtype=torch.float16,
    cache_dir=TMP_CACHE,
)
pipe.save_pretrained(os.path.join(MODEL_CACHE, 'cyberrealisticv42'))

shutil.rmtree(TMP_CACHE)