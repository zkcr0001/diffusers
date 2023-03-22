from base64 import b64encode
import os

import numpy
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline, UniPCMultistepScheduler
from huggingface_hub import notebook_login

# For video display:
from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import numpy as np
import torchvision.transforms.functional as F

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')

def main():
  # Supress some unnecessary warnings when loading the CLIPTextModel
  logging.set_verbosity_error()

  # Set device
  torch_device = "cuda" if torch.cuda.is_available() else "cpu"

  # Load hed images

  root_path = "/home/ubuntu/AI二次元/0319_jiaran_yao"
  hed_img_path = os.path.join(root_path, "transfered_images")
  file_list = os.listdir(hed_img_path)
  img_num = len(file_list)
  img_num = 100
  hed_img_list = []
  for i in range(img_num):
      filename = str(i+1) + ".png"
      file_path = os.path.join(hed_img_path, filename)
      hed_image = Image.open(file_path)
      hed_image = SquarePad()(hed_image)
      hed_image = F.resize(hed_image, size=(512,512))   
      hed_img_list.append(hed_image)
  fig, ax = plt.subplots()
  ax.imshow(hed_image)
  
  # Generate random noise
  height = 512              
  width = 512 
  batch_size = 1

  noisy_img_list = []

  for i in range(img_num):
      # Prep latents
      torch.manual_seed(32)
      latents = torch.randn(
        (batch_size, 4, height // 8, width // 8),
      )
      noisy_img_list.append(latents)
  print(torch.allclose(noisy_img_list[0],noisy_img_list[5]))

  # Test Controlnet Pipeline
  controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
  pipe = StableDiffusionControlNetPipeline.from_pretrained(
      "gsdf/Counterfeit-V2.5", controlnet=controlnet, torch_dtype=torch.float16
  )
  # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  pipe.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
  pipe = pipe.to("cuda")
  
  torch.manual_seed(32)
  prompt = "masterpiece, best quality, 1girl, solo"
  negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
  output = pipe(
      prompt,
      hed_img_list[0],
      negative_prompt=negative_prompt,
      num_inference_steps=20,
  )
  plt.figure(figsize=(10,10))
  fig, ax = plt.subplots()
  ax.imshow(output[0][0])

  # Decompose controlnet
  vae = pipe.vae.to("cuda")
  tokenizer = pipe.tokenizer
  text_encoder = pipe.text_encoder.to("cuda")
  unet = pipe.unet.to("cuda")
  scheduler = pipe.scheduler
  controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16).to("cuda")
  print("")


if __name__ == "__main__":
    main()