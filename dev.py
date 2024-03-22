from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DPMSolverMultistepScheduler, AutoencoderKL
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image
import cv2

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter", 
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
).to("cuda")

image = load_image("https://replicate.delivery/pbxt/KbJQPaj3KBHz0kuaddny3Wf4ZhQpYyfDaJhecL7CebigvphB/input_1176.png")

image_orig = image

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)

control_image.save("control.png")

controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16
)

controlnet_tile = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile",
    torch_dtype=torch.float16
)

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",  
    torch_dtype=torch.float16,
).to("cuda")

pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
    "https://huggingface.co/philz1337x/cyberrealistic-v4.2/cyberrealistic_v42.safetensors",
    image_encoder = image_encoder, 
    controlnet=controlnet_canny, 
    safety_checker=None,
    torch_dtype=torch.float16,
    vae=vae
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

ip_image = load_image("https://firebasestorage.googleapis.com/v0/b/upscaler-89296.appspot.com/o/inputs%2Fmzy66feAaaQ5fdkZzdrobsXM9W63-5852ea59-49dd-4d57-b607-837cce976b93.jpg?alt=media&token=35363349-224d-4ffb-8dde-3f2be70f7152")

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.4)

pipe.enable_model_cpu_offload()


generator = torch.manual_seed(1337)

image = pipe(
    prompt="masterpiece, best quality, highres, wallpaper,", 
    image=image_orig,
    control_image=control_image,
    negative_prompt="(low quality, bad quality, worst quality:1.2)", 
    ip_adapter_image=ip_image, 
    num_inference_steps=50,
    guidance_scale=8,
    controlnet_conditioning_scale=0.6,
    guess_mode=False,
    strength=1.0,
    generator=generator,
).images[0]

image.save('canny_and_style.png')