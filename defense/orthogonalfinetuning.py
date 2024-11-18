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

class OrthogonalFineTuner:
    def __init__(self, model, device='cuda', lr=1e-4, num_steps=100):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.num_steps = num_steps

        self.proj_weights = {}
        self.A_matrices = {}
        self.C_matrices = {}

        layers_to_finetune = [
                "model.proj",                        # Projects extracted features to embedding space
                "model.transformer.resblocks.0.attn.in_proj_weight",  # Attention projection weight in the first transformer block
                "model.transformer.resblocks.0.attn.out_proj.weight", # Output projection weight for attention in the first block
                "model.transformer.resblocks.0.mlp.c_fc.weight",       # Fully connected layer weight in the first block’s MLP
                "model.transformer.resblocks.0.mlp.c_proj.weight",
            ]
        self.layers_to_finetune = layers_to_finetune
        self.num_layers = len(layers_to_finetune)
        for name, param in self.model.named_parameters():
            if name in layers_to_finetune:
                self.proj_weights[name] = param.clone().to(torch.float32).to(self.device)
                C_matrix = torch.zeros(param.shape[0],param.shape[0], requires_grad=True, device = self.device)
                I = torch.eye(C_matrix.shape[0], device=self.device)
                A = ((I + C_matrix).inverse() @ (I - C_matrix)).requires_grad_(True)
                self.C_matrices[name] = C_matrix                
                self.A_matrices[name] = A

        self.optimizer = torch.optim.Adam(self.C_matrices.values(), lr=self.lr)
        self.losses = []

    def compute_hyperspherical_energy(self, W, num_neighbors=10):
        W = W.to(self.device)  # Ensure W is on the correct device
        W_normalized = W / W.norm(dim=0, keepdim=True)
        num_vectors = W.shape[1]
        energy = torch.tensor(0.0, device=self.device)  # Initialize on device
        similarity_matrix = W_normalized.T @ W_normalized  # Shape: (num_vectors, num_vectors)
        distances = 2 - 2 * similarity_matrix  # Distance formula for normalized vectors
        distances += torch.eye(num_vectors, device=self.device) * 1e-6  # Ensures self-distance is non-zero
        for i in range(num_vectors):
            vector_distances = distances[i]
            nearest_distances, _ = torch.topk(1.0 / vector_distances, k=num_neighbors + 1)
            energy += torch.sum(nearest_distances[1:])
        energy /= num_vectors
        return energy

    def orthogonalize_step(self):
        total_loss = torch.tensor(0.0, device=self.device)  # Initialize on device
        for name, W0 in self.proj_weights.items():
            W0 = W0.to(self.device)  # Ensure W0 is on the correct device
            C = self.C_matrices[name]
            I = torch.eye(C.shape[0], device=self.device)
            A = (I + C).inverse() @ (I - C)
            new_weights = A @ W0

            # Update the model's weights with the new transformed weights
            # Uncomment if you want to update weights directly in the model
            # for model_name, param in self.model.named_parameters():
            #     if model_name == name:
            #         param.data = new_weights.to(torch.float16).data

            hyperspherical_energy = self.compute_hyperspherical_energy(new_weights)
            pretrained_energy = self.compute_hyperspherical_energy(W0)
            loss = torch.abs(hyperspherical_energy - pretrained_energy).to(self.device)
            total_loss += loss

        return total_loss / self.num_layers

    def finetune(self, loss):
        # Fine-tuning loop
        for step in range(self.num_steps):
            self.optimizer.zero_grad()
            
            # Ensure the loss is on the correct device
            loss = loss.to(torch.float32).to(self.device).requires_grad_(True)
            
            # Backward and update C
            loss.backward()
            self.optimizer.step()

            # Store the loss
            self.losses.append(loss.item())
        
        return self.model
