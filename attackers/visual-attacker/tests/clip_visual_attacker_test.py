import torch
import clip
from PIL import Image
import requests
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_attacker import Attacker
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Create output directory if it doesn't exist
output_dir = "plots"
num_iter = 11
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
image = preprocess(Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)).unsqueeze(0).to(device)

# Set target and class names
target = torch.tensor([0]).to(device)
classes = ["a car", "a dog", "a cat", "Yassine Guennoun"]

# Initialize the attacker
attacker = Attacker(model, classes, device=device, eps=8/255)

# Perform specific attack
adv_image_specific, losses_specific = attacker.attack_specific(image, target, num_iter=num_iter)
model_output = torch.tensor([2]).to(device)
# Perform unspecific attack
adv_image_unspecific, losses_unspecific = attacker.attack_unspecific(image, model_output,num_iter=num_iter)
print(attacker.generate_prompt(adv_image_specific.to(device)))
print(attacker.generate_prompt(adv_image_unspecific.to(device)))
# Plot and save the loss values for the specific attack
plt.figure(figsize=(10, 6))
plt.plot(losses_specific, label="Specific Attack Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss During Specific Adversarial Attack")
plt.legend()

# Save the specific attack loss plot
loss_plot_file_specific = os.path.join(output_dir, "loss_plot_specific.png")
plt.savefig(loss_plot_file_specific)
print(f"Specific attack loss plot saved to {loss_plot_file_specific}")

# Plot and save the loss values for the unspecific attack
plt.figure(figsize=(10, 6))
plt.plot(losses_unspecific, label="Unspecific Attack Loss", color='red')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss During Unspecific Adversarial Attack")
plt.legend()

# Save the unspecific attack loss plot
loss_plot_file_unspecific = os.path.join(output_dir, "loss_plot_unspecific.png")
plt.savefig(loss_plot_file_unspecific)
print(f"Unspecific attack loss plot saved to {loss_plot_file_unspecific}")
