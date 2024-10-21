import torch
import clip
from PIL import Image
import requests
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToPILImage
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_attacker import Attacker
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import kornia.filters as kf

# Create directory to save the results
output_dir = "defense_results"
os.makedirs(output_dir, exist_ok=True)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
num_iter = 11

# Download the CIFAR-100 dataset and pick consistent images
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
n_samples = 50  # Fixed number of images
fixed_indices = list(range(n_samples))  # Use first 50 images (fixed)
images_and_labels = [(cifar100[i][0], cifar100[i][1]) for i in fixed_indices]
target_class_id = 0
batch_size = 2

#we load the normal image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url=url, stream=True).raw)).unsqueeze(0).to(device)

targets = torch.tensor([0]).to(device)
# DISCLAIMER : These offensive words are used for research purposes only, to demonstrate the lack of robustness of the model
classes = ["a car", "a dog", "a cat", "Yassine Guennoun"]

attacker = Attacker(model, classes, device = device, eps=8/255)

adv_image, losses = attacker.attack_specific(image, targets, num_iter=num_iter)


# finding optimal value of d and sigma

sigma_values = [x/20 for x in range(1, 100)]
loss_values=[]
result = []
cat_proba = []
car_proba = []

for sig in sigma_values:
    adversarial_image = adv_image.to(device)

    denoised_image = kf.bilateral_blur(adversarial_image, kernel_size=(5, 5), sigma_color=sig, sigma_space=(sig,sig))
    
    print(sig)

    loss_values.append(attacker.loss(denoised_image, targets).item())

    result.append(attacker.predict(denoised_image))
    cat_proba.append(attacker.proba_vect(denoised_image)[0, 2].item())
    car_proba.append(attacker.proba_vect(denoised_image)[0, 0].item())

correct_predictions = [pred == 'a cat' for pred in result]

# plot the optimization of sigma

plt.figure(figsize=(10, 6))
plt.scatter(
    [sigma_values[i] for i in range(len(sigma_values)) if correct_predictions[i]], 
    [loss_values[i] for i in range(len(loss_values)) if correct_predictions[i]], 
    color='green', label='Correct (cat)', s=100
)
plt.scatter(
    [sigma_values[i] for i in range(len(sigma_values)) if not correct_predictions[i]], 
    [loss_values[i] for i in range(len(loss_values)) if not correct_predictions[i]], 
    color='red', label='Incorrect', s=100
)
plt.xlabel("Sigma")
plt.ylabel("Loss")
plt.title("Loss Depending on Sigma")
plt.legend()

# Save the sigma plot
plt.savefig(os.path.join(output_dir, "sigma_plot.png"))
plt.close()
print(f"Sigma plot saved to {output_dir}")

# plot the evolution of probability during optimization

plt.figure(figsize=(10, 6))

plt.plot(sigma_values, cat_proba, color='b', label='proba cat')
plt.plot(sigma_values, car_proba, color='r', label='proba car')
plt.xlabel("Sigma")
plt.ylabel("Proba")
plt.title("Proba Depending on Sigma")
plt.legend()

# Save the proba plot
plt.savefig(os.path.join(output_dir, "proba_plot.png"))
plt.close()
print(f"Proba plot saved to {output_dir}")