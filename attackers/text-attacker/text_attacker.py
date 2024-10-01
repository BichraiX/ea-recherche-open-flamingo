import torch
from tqdm import tqdm
import clip
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
    
def generate_prompt(model,classes,image,text):
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
    return classes[logits_per_image.argmax().item()]
        

class Attacker:
    def __init__(self,model,targets,classes, device='cuda:0', eps = 1/255, alpha = 0.01):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target = targets.to(self.device)
        self.eps = eps
        self.classes = classes
        self.text = clip.tokenize(self.classes).to(self.device)

        self.alpha = alpha
        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

        
    def attack(self, img, batch_size = 8, num_iter = 2000):
        adv_noise = torch.randn_like(img).to(self.device) * 2 * self.eps - self.eps
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data # Clamp values to be between 0 and 1
        adv_noise.requires_grad = True
        adv_noise.retain_grad()

        # Initialize list to store loss values
        loss_values = []

        for t in tqdm(range(num_iter)):
            x_adv = normalize(x + adv_noise)
            logits_per_image, logits_per_text = self.model(x_adv, self.text)            
            # Calculate loss and append it to the loss list
            target_loss = torch.nn.functional.cross_entropy(logits_per_image, self.target)
            loss_values.append(target_loss.item())
            
            target_loss.backward()
            adv_noise.data = (adv_noise.data - self.alpha * adv_noise.grad.detach().sign()).clamp(-self.eps, self.eps)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data            
            adv_noise.grad.zero_()
            self.model.zero_grad()
            
            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                print(logits_per_image)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)

                with torch.no_grad():
                    print('>>> Sample Outputs')
                    print(generate_prompt(self.model, self.classes, x_adv, self.text))
                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)

        return adv_img_prompt, loss_values
