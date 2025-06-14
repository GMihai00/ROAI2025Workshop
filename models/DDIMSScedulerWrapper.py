from diffusers import DDIMScheduler 

class DDIMSScedulerWrapper:
    def __init__(self):
        self.scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", use_auth_token=True)

    def add_initial_noise(self, latents, iteration):
        # DDIMScheduler does not require manual noise addition
        return latents
        
    def get_timesteps(self):
        return self.scheduler.timesteps
        
    def set_timesteps(self, num_inference_steps):
        self.scheduler.set_timesteps(num_inference_steps)
    
    def add_noise(self, latents, noise, timestamp):
        self.scheduler.add_noise(latents, noise, timestamp)
        
    def add_noise_inference(self, latent, iteration):
        # automatically handled by the scheduler
        pass
        
    def step(self, noise_pred, timestamp, latents):
        return self.scheduler.step(noise_pred, timestamp, latents)