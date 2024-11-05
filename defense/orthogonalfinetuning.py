import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToPILImage
from visual_attacker import Attacker
from defense import RandomizedSmoothing
import time

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

class OrthogonalFineTuner:
    def __init__(self, model, device='cuda', lr=1e-4, num_steps=100):
        self.model = model
        self.device = device
        self.lr = lr
        self.num_steps = num_steps
        
        # Set the model to evaluation mode and freeze weights
        self.model.eval()
        self.model.requires_grad_(False)

        # Store the original weights of ln_final
        for name, param in self.model.named_parameters():
            if name == 'transformer.resblocks.9.attn.in_proj_weight':
                self.W0 = (param.clone()).to(torch.float32)

        # Initialize anti-symmetric matrix C
        self.C = torch.zeros(self.W0.shape[0], self.W0.shape[0], device=self.device, requires_grad=True)
        
        # Optimizer for C
        self.optimizer = torch.optim.Adam([self.C], lr=self.lr)

        # List to store losses
        self.losses = []

    def compute_hyperspherical_energy(self, W):
        W_normalized = W / W.norm(dim=0, keepdim=True)
        energy = sum((W_normalized[:, i] - W_normalized[:, j]).norm()**-1 for i in range(W.shape[1]) for j in range(i + 1, W.shape[1]))
        return energy

    def orthogonalize_step(self):
        #for name, param in self.model.named_parameters():
         #   print(name)  # Afficher le nom de la couche

        # Compute the orthogonal matrix A from the anti-symmetric matrix C
        I = torch.eye(self.C.shape[0], device=self.device)
        A = (I + self.C).inverse() @ (I - self.C)
        # Apply the orthogonal transformation to ln_final weights
        new_ln_final_weights = (A @ self.W0)

        # Update ln_final weights with the new transformed weights
        for name, param in self.model.named_parameters():
            if name == 'transformer.resblocks.9.attn.in_proj_weight':
                param.data = new_ln_final_weights.to(torch.float16).data
        
        # Compute the loss
        hyperspherical_energy = self.compute_hyperspherical_energy(new_ln_final_weights)
        pretrained_energy = self.compute_hyperspherical_energy(self.W0)
        loss = torch.abs(hyperspherical_energy - pretrained_energy)

        return loss

    def finetune(self):
        # Fine-tuning loop
        for step in range(self.num_steps):
            self.optimizer.zero_grad()
            
            # Forward pass and compute loss
            loss = self.orthogonalize_step()
            loss = loss.to(torch.float32).requires_grad_(True)
            
            # Backward and update C
            loss.backward()
            self.optimizer.step()

            # Store the loss
            self.losses.append(loss.item())

            if step % 1 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
        
        return self.model

# Chargement du modèle CLIP
model, _ = clip.load("ViT-B/32", device="cuda")

for name,param in model.named_parameters():
    if name=='transformer.resblocks.9.attn.in_proj_weight':
        old_weights = param.data
# Créer une instance du fine-tuner orthogonal
fine_tuner = OrthogonalFineTuner(model)

save_path = './output'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = fine_tuner.finetune()

for name,param in model.named_parameters():
    if name=='transformer.resblocks.9.attn.in_proj_weight':
        print(old_weights)
        print(param.data)
        print("The weights have changed : ", old_weights == param.data)
# Plot the loss
plt.figure(figsize=(10, 5))
plt.plot(fine_tuner.losses, label='Loss', color='blue')
plt.title('Loss during Fine-Tuning')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
