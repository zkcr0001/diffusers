import numpy as np
import torch
from PIL import Image
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from diffusers import ControlNetModel, UniPCMultistepScheduler, LMSDiscreteScheduler
from diffusers.utils import load_image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from pathlib import Path
import albumentations

input_image = Image.open("/home/ubuntu/input_toys/0001.png")
input_image.thumbnail((512,512))
image = np.array(input_image)

low_threshold = 100
high_threshold = 200

canny_image = cv2.Canny(image, low_threshold, high_threshold)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image = Image.fromarray(canny_image)

import torch
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

pipe_controlnet = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    safety_checker=None, 
    torch_dtype=torch.float16
)

# pipe_controlnet.scheduler = UniPCMultistepScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
# pipe_controlnet.enable_xformers_memory_efficient_attention()
pipe_controlnet.enable_model_cpu_offload()

# using image with edges for our canny controlnet

result_img = pipe_controlnet(controlnet_conditioning_image=canny_image, 
                        image=input_image,
                        prompt="a toy, a blue background", 
                        num_inference_steps=50,  strength = 1, controlnet_conditioning_scale = 0.5).images[0]
result_img