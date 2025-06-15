from models.ImageGenerator import ImageGenerator
from models.LMSDiscreteSchedulerWrapper import LMSDiscreteSchedulerWrapper
from models.DDIMSScedulerWrapper import DDIMSScedulerWrapper

import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = DDIMSScedulerWrapper()

imageGenerator = ImageGenerator(scheduler=scheduler, torch_device=torch_device)

# prompt = ["""A student. While participating in the olympia of informatics learnt the truths of AI and is now 
# now rapidly gaining knowledge over the world with the help of AI. 4k, detailed, trending in artstation, fantasy vivid colors. 
# He is sitting down and using it's laptop with a big smile on his face. His face is towards the camera and is looking happy."""]
prompt = ["A disney princess in a fantasy world, 4k, detailed, trending in artstation, fantasy vivid colors"]
image_path = "./rabbit.jpg"
height = 512
width = 512
start_step = 25
num_inference_steps = 75
guidance_scale = 7.5
batch_size = 1
# for reproducibility, to uniquely generated the same image every time
generator = torch.manual_seed(4)


images = imageGenerator.augment_image(prompt=prompt,
                        image_path=image_path,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        batch_size=batch_size,
                        generator=generator)

images[0].show()

