from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DPMSolverMultistepScheduler, AutoencoderKL
from PIL import Image
import numpy as np
import torch
import cv2
import os
import shutil

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def scale_image(self, image):
        width, height = image.size
        scale = (780000 / (width * height)) ** 0.5
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        if new_width % 64 != 0:
            new_width = ((new_width // 64) + 1) * 64
        
        new_height = int((new_width / width) * height)
        
        return image.resize((new_width, new_height))

    def load_image(self, image_path: Path):
        if image_path is None:
            return None
        if os.path.exists("img.png"):
            os.unlink("img.png")
        shutil.copy(image_path, "img.png")
        img = Image.open("img.png")
        return img
    
    def upscale(self, img, upscale_rate=1):
        w, h = img.size
        new_w, new_h = int(w * upscale_rate), int(h * upscale_rate)
        return img.resize((new_w, new_h), Image.BICUBIC)
    
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", 
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to("cuda")
        
        self.controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers-cache/canny",
            torch_dtype=torch.float16
        ).to("cuda")

        self.controlnet_tile = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            torch_dtype=torch.float16
        ).to("cuda")
        
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",  
            torch_dtype=torch.float16,
        ).to("cuda")

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "diffusers-cache/cyberrealisticv42",
            image_encoder = self.image_encoder, 
            controlnet=self.controlnet_canny, 
            safety_checker=None,
            torch_dtype=torch.float16,
            vae=self.vae,
            local_files_only=False
        ).to("cuda")

        self.pipe_tile = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "diffusers-cache/cyberrealisticv42",
            image_encoder = self.image_encoder, 
            controlnet=self.controlnet_tile, 
            safety_checker=None,
            torch_dtype=torch.float16,
            vae=self.vae,
            local_files_only=False
        ).to("cuda")
        
        print(f"Setup finished.")

    def predict(
        self,
        image: Path = Input(description="input image"),
        image_style: Path = Input(description="image for style"),
        style_strength: float = Input(
            description="How much the style should get applied", ge=0, le=3, default=0.4
        ),
        structure_strength: float = Input(
            description="How much the structure should keep the same", ge=0, le=3, default=0.6
        ),
        prompt: str = Input(
            description="Prompt", default="masterpiece, best quality, highres"
        ),
        negative_prompt: str = Input(
            description="Negative Prompt", default="worst quality, low quality, normal quality"
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=30
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=8
        ),
        seed: int = Input(
            description="Leave blank to randomize the seed", default=1337
        )
    ) -> list[Path]:
        """Run a single prediction on the model"""
        image = self.load_image(image)
        image = self.scale_image(image)
        image_orig = image

        image_style = self.load_image(image_style)
        image_style = self.scale_image(image_style)

        image = np.array(image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.use_karras_sigmas = True

        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        generator = torch.manual_seed(seed)

        self.pipe.set_ip_adapter_scale(style_strength)

        img2img_strength = 1.0
        if style_strength < 0.3:
            img2img_strength = style_strength * 2

        output = self.pipe(
            prompt=prompt, 
            image=image_orig,
            control_image=control_image,
            negative_prompt=negative_prompt, 
            ip_adapter_image=image_style, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=structure_strength,
            guess_mode=False,
            strength=img2img_strength,
            generator=generator,
        )

        self.pipe_tile.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        self.pipe_tile.set_ip_adapter_scale(style_strength)

        output = self.pipe_tile(
            prompt=prompt, 
            image=output.images[0],
            control_image=output.images[0],
            negative_prompt=negative_prompt, 
            ip_adapter_image=image_style, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=structure_strength,
            guess_mode=False,
            strength=img2img_strength,
            generator=generator,
        )

        output_paths = []
        for i, nsfw in enumerate(output):
            output_path = f"/tmp/out-{str(seed)}-{i}.png"
            output.images[i].save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )
        
        return output_paths