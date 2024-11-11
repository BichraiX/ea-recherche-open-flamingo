import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToPILImage
from attackers.visual_attacker.visual_attacker import Attacker
from defense.defense import RandomizedSmoothing
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
            if name == 'model.transformer.resblocks.11.mlp.c_proj.weight':
                self.W0 = (param.clone()).to(torch.float32)

        # Initialize anti-symmetric matrix C
        self.C = torch.zeros(self.W0.shape[0], self.W0.shape[0], device=self.device, requires_grad=True)
        
        # Optimizer for C
        self.optimizer = torch.optim.Adam([self.C], lr=self.lr)

        # List to store losses
        self.losses = []

    
    def compute_hyperspherical_energy(self, W, num_neighbors=10):
        # Normalize W along columns
        W_normalized = W / W.norm(dim=0, keepdim=True)
        num_vectors = W.shape[1]
        energy = 0.0

        # Compute the pairwise dot products
        similarity_matrix = W_normalized.T @ W_normalized  # Shape: (num_vectors, num_vectors)

        # Convert dot products to Euclidean distances efficiently
        distances = 2 - 2 * similarity_matrix  # Distance formula for normalized vectors

        # Add a small epsilon to avoid division by zero in the next step
        distances += torch.eye(num_vectors, device=W.device) * 1e-6  # Ensures self-distance is non-zero

        # Loop over each vector to accumulate energy based on nearest neighbors
        for i in range(num_vectors):
            # Get distances for the i-th vector to all others
            vector_distances = distances[i]
            
            # Sort and select the nearest distances (excluding itself)
            nearest_distances, _ = torch.topk(1.0 / vector_distances, k=num_neighbors + 1)
            
            # Accumulate energy, ignoring the first item (distance to itself)
            energy += torch.sum(nearest_distances[1:])

        # Scale energy by 1 / num_vectors to maintain proportionality and gradient scale
        energy /= num_vectors
        return energy


    def orthogonalize_step(self):
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

