from models.ImageGenerator import ImageGenerator
from models.LMSDiscreteSchedulerWrapper import LMSDiscreteSchedulerWrapper
from models.DDIMSScedulerWrapper import DDIMSScedulerWrapper

import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# scheduler = LMSDiscreteSchedulerWrapper()
scheduler = DDIMSScedulerWrapper()

imageGenerator = ImageGenerator(scheduler=scheduler, torch_device=torch_device)

prompt = ["""A student. While participating in the olympia of informatics learnt the truths of AI and is now 
now rapidly gaining knowledge over the world with the help of AI. 4k, detailed, trending in art station, fantasy vivid colors. 
He is sitting down and using it's laptop. His face is towards the camera. The image is bright and realistic, with a focus on the student's expression and the laptop screen showing complex algorithms and data visualizations."""]

# prompt = ["A digital illustration of a steampunk computer laboratory with clockwork machines, 4k, detailed, trending in artstation, fantasy vivid colors"]
height = 512
width = 768
num_inference_steps = 50
guidance_scale = 7.5
batch_size = 1
# for reproducibility, to uniquely generated the same image every time

for i in range(0, 5):
    print(f"Generating image {i+1}...")
    generator = torch.manual_seed((i + 1) * 5)
    
    images = imageGenerator.generate(prompt=prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            batch_size=batch_size,
                            generator=generator)
    
    # images[0].show()
    images[0].save(f"generated_image_{i+1}.png")
