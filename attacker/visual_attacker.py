import torch
from tqdm import tqdm
import random
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import seaborn as sns

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

class Attacker:
    def __init__(self,model,targets, device='cuda:0', eps = 1/255):
        self.model = model
        self.targets = targets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.num_targets = len(targets)
        
        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

        
    def attack(self, img, tokenizer, batch_size = 8, num_iter = 2000,):
        assert len(img.shape) == 4, "Input image should be a torch tensor of shape 1x1xchannels x height x width"
        # We expect the input image to be a torch tensor of shape 1 x 1 xchannels x height x width since we use images not videos
        # OpenFlamingo expects the image to be a torch tensor of shape batch_size x num_media x num_frames x channels x height x width
        adv_noise = torch.randn_like(img).to(self.device) * 2 * self.eps - self.eps
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data # We need to clamp the values to be between 0 and 1
        adv_noise.requires_grad = True
        adv_noise.retain_grad()
        
        for i in tqdm(range(num_iter)):
            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size
            # We need to repeat the image to match the batch size so that it matches the input shape of the model
            x_adv = normalize(adv_noise).repeat(batch_size, 1, 1, 1, 1, 1)
            lang_x = tokenizer(
                ["<image><|endofchunk|>"], # empty text prompt for just an image
                return_tensors="pt",)
            # generated_text = self.model.generate(vision_x = x_adv,lang_x = lang_x["input_ids"], attention_mask = lang_x["attention_mask"]) # We use OpenFlamingo's generate function
            ## TODO : Analyze what num_beams and max_new_tokens do
            output = self.model(
                vision_x=x_adv,                  # Adversarially perturbed images
                lang_x=lang_x["input_ids"],      # Tokenized text inputs
                attention_mask=lang_x["attention_mask"],  # Attention mask for text
                labels=tokenized_targets         # Tokenized target texts for loss computation
            )
            
            # Compute the loss and backpropagate
            target_loss = output.loss
            print("target_loss : ", target_loss)
            print("output : ", output)
            break
        return generated_text
            
            
        