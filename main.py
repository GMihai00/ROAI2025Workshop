from models.ImageGenerator import ImageGenerator
from models.LMSDiscreteSchedulerWrapper import LMSDiscreteSchedulerWrapper
from models.DDIMSScedulerWrapper import DDIMSScedulerWrapper

import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = LMSDiscreteSchedulerWrapper()
# scheduler = DDIMSScedulerWrapper()

imageGenerator = ImageGenerator(scheduler=scheduler, torch_device=torch_device)

prompt = ["A digital illustration of a steampunk computer laboratory with clockwork machines, 4k, detailed, trending in artstation, fantasy vivid colors"]
height = 512
width = 768
num_inference_steps = 50
guidance_scale = 7.5
batch_size = 1
# for reproducibility, to uniquely generated the same image every time
generator = torch.manual_seed(4)

images = imageGenerator.generate(prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        batch_size=batch_size,
                        generator=generator)

images[0].show()

