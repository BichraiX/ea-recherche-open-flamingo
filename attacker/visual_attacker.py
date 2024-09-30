import torch
from tqdm import tqdm
import random
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import seaborn as sns

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

        
    def attack(self, img, batch_size = 32, num_iter = 2000):
        adv_noise = torch.randn_like(img).to(self.device)
        adv_noise.requires_grad = True
        adv_noise.retain_grad()
        
        for i in tqdm(range(num_iter)):
            batch_targets = random.sample(self.targets, batch_size)
            
        