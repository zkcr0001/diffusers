import numpy as np
import requests
import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from diffusers import CycleDiffusionPipeline, DDIMScheduler

# load the pipeline
# make sure you're logged in with `huggingface-cli login`
model_id_or_path = "gsdf/Counterfeit-V2.5"
scheduler = DDIMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
pipe = CycleDiffusionPipeline.from_pretrained(model_id_or_path, scheduler=scheduler).to("cuda",torch.float64)

# let's download an initial image
init_image = Image.open("/home/ubuntu/anime_dance/0001.png").convert("RGB")
init_image = init_image.resize((512, 512))

torch.manual_seed(11)
# let's specify a prompt
source_prompt = "masterpiece,best quality, 1girl, long hair, red hair, solo, dress, blue eyes, looking at viewer"
prompt = "masterpiece,best quality, 1girl, long hair, red hair, solo, dress, blue eyes, looking at viewer"

# call the pipeline
image1 = pipe(
    prompt=prompt,
    source_prompt=source_prompt,
    image=init_image,
    num_inference_steps=20,
    eta=0.1,
    strength=0.8,
    guidance_scale=7.5,
    source_guidance_scale=7.5,
).images[0]