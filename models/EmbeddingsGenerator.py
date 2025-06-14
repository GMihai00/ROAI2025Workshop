from transformers import CLIPTextModel, CLIPTokenizer
import torch

class EmbeddingsGenerator:
    def __init__(self, torch_device="cuda"):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", use_auth_token=True)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", use_auth_token=True)
        self.torch_device = torch_device
        
        if torch_device == "cuda":
            self.text_encoder = self.text_encoder.to(torch_device).half()

    def encode(self, prompt, batch_size):
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        
        # blank text
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        return text_embeddings

