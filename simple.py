
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
prompt = "A futuristic city skyline at sunset"
image = pipe(prompt, num_inference_steps=50).images[0]

image.show()
# display(image)
image.save("output.png")

