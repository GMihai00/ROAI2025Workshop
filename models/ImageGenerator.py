from diffusers import  UNet2DConditionModel
import torch
from tqdm.auto import tqdm

from models.ImageHelper import ImageHelper
from models.EmbeddingsGenerator import EmbeddingsGenerator
from PIL import Image
class ImageGenerator:
    def __init__(self, scheduler, torch_device="cuda", unet=None):
        
        self.scheduler = scheduler

        if unet is not None:
            self.unet = unet
        else:
            self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True)
            
        self.image_helper = ImageHelper(torch_device=torch_device)
        self.embeddings_generator = EmbeddingsGenerator(torch_device=torch_device)
        self.torch_device = torch_device
        if  torch_device == "cuda":
            self.unet = self.unet.to(torch_device).half()
    
    def prepare_initial_latent_noise(self, batch_size, height, width, generator, start_step = 0, starting_latent=None):
        
        if starting_latent is None:
            starting_latent = torch.randn(
                (batch_size, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
        else:
            # noise the initial image
            start_timestep = self.scheduler.get_timesteps()[start_step]
            noise = torch.randn_like(starting_latent)
            latents = self.scheduler.add_noise(starting_latent, noise, start_timestep)
        
        latents = starting_latent.to(self.torch_device).half()
        latents = self.scheduler.add_initial_noise(latents=latents, iteration=start_step)
        return latents
        
    def predict_noise(self, latents, timestamp, text_embeddings, guidance_scale):
        with torch.no_grad():
            noise_pred = self.unet(latents, timestamp, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # in here we specify the level of guidance we should use
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        return noise_pred
    
    def reconstruct_previous_latent(self, noise_pred, timestamp, latents):
        latents = self.scheduler.step(noise_pred, timestamp, latents)["prev_sample"]
        return latents
    
    
    def augment_image(self, image_path,
        prompt, 
        height = 512, width = 768, 
        start_step = 25,
        num_inference_steps = 50, 
        guidance_scale = 7.5, 
        batch_size = 1 ,
        generator = torch.manual_seed(4)): 
        
        im = Image.open(image_path).convert('RGB')
        im = im.resize((height,width))
        encoded = self.image_helper.pil_to_latent(im)
        
        images = self.generate(prompt, height=height, width=width,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale, 
            batch_size=batch_size,
            generator=generator,
            start_step=start_step,
            starting_latent=encoded)
            
        return images
    
    def generate(self, prompt, 
        height = 512, width = 768, 
        num_inference_steps = 50, 
        guidance_scale = 7.5, 
        batch_size = 1 ,
        generator = torch.manual_seed(4),
        start_step = 0,
        starting_latent=None):
        
        text_embeddings = self.embeddings_generator.encode(prompt=prompt, batch_size=batch_size)  
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        latents = self.prepare_initial_latent_noise(batch_size, height, width, generator, start_step, starting_latent)
        
        with torch.autocast(self.torch_device):
            for i, t in tqdm(enumerate(self.scheduler.get_timesteps())):
                if i >= start_step:
                    # expand the latents to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    
                    self.scheduler.add_noise_inference(latent_model_input, i)
                    
                    noise_pred = self.predict_noise(latent_model_input, t, text_embeddings, guidance_scale)
            
                    latents = self.reconstruct_previous_latent(noise_pred, t, latents)
                
        images = self.image_helper.latents_to_pil(latents)
        
        return images
        
            
        