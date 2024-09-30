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
        
    def attack(self, text_prompt_template, batch_size = 32, num_iter = 2000):
        