from diffusers import AutoencoderKL
from torchvision import transforms as tfms
import torch
from PIL import Image

class ImageHelper:
    def __init__(self, torch_device="cuda"):
        self.to_tensor_tfm = tfms.ToTensor()
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True)
        self.torch_device = torch_device
        if torch_device == "cuda":
            self.vae = self.vae.to(torch_device).half()
        
    def pil_to_latent(self, input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        tensor = self.to_tensor_tfm(input_im).unsqueeze(0).to(self.torch_device) * 2 - 1  # Note scaling
        if self.torch_device == "cuda":
            tensor = tensor.half()
        with torch.no_grad():
            encoded = self.vae.encode(tensor)
            latent = encoded.latent_dist.mean # or .mean or .sample
        return 0.18215 * latent 

    def latents_to_pil(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


