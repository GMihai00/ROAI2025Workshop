from models.ImageGenerator import ImageGenerator
from models.LMSDiscreteSchedulerWrapper import LMSDiscreteSchedulerWrapper
from models.DDIMSScedulerWrapper import DDIMSScedulerWrapper

import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = LMSDiscreteSchedulerWrapper()
# scheduler = DDIMSScedulerWrapper()

imageGenerator = ImageGenerator(scheduler=scheduler, torch_device=torch_device)

# prompt = ["A person that is turning into a cyborg, 4k, trending in artstation, fantasy vivid colors"]
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


# prompt = ["A person that is turning into a cyborg, 4k, trending in artstation, fantasy vivid colors"]
# image_path = "./WhatsApp Image 2025-06-13 at 17.32.14.jpeg"
# height = 512
# width = 512
# start_step = 50
# num_inference_steps = 500
# guidance_scale = 7.5
# batch_size = 1
# # for reproducibility, to uniquely generated the same image every time
# generator = torch.manual_seed(4)


# images = imageGenerator.augment_image(prompt=prompt,
#                         image_path=image_path,
#                         height=height,
#                         width=width,
#                         num_inference_steps=num_inference_steps,
#                         guidance_scale=guidance_scale,
#                         batch_size=batch_size,
#                         generator=generator)

# images[0].show()


