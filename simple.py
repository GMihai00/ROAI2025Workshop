
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from IPython.display import display 

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {torch_device}")

# Load the DDIM scheduler
ddim_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", use_auth_token=True)

# Load the pipeline with custom scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    scheduler=ddim_scheduler,
    torch_dtype=torch.float16  # Optional: for faster inference if using GPU
).to("cuda")

# Run inference
prompt = """A student face. While participating in the olympia of informatics learnt the truths of AI and is now 
now rapidly gaining knowledge over the world with the help of AI."""


for i in range(0, 2):
    
    image = pipe(prompt, num_inference_steps=50).images[0]
    
    image.show()
    
    image.save(f"./images/output{i}.png")

