
from diffusers import  LMSDiscreteScheduler

class LMSDiscreteSchedulerWrapper():
    def __init__(self):
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    def add_initial_noise(self, latents, iteration):
        return latents * self.scheduler.sigmas[iteration]
        
    def get_timesteps(self):
        return self.scheduler.timesteps
        
        
    def set_timesteps(self, num_inference_steps):
        self.scheduler.set_timesteps(num_inference_steps)
        
    def add_noise(self, latents, noise, timestamp):
        self.scheduler.add_noise(latents, noise, timestamp)
        
    def add_noise_inference(self, latent, iteration):
        sigma = self.scheduler.sigmas[iteration]
        latent = latent / ((sigma**2 + 1) ** 0.5)
        return latent
        
    def step(self, noise_pred, timestamp, latents):
        return self.scheduler.step(noise_pred, timestamp, latents)
        