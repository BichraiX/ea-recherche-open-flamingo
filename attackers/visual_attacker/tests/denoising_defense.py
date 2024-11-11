import torch
import clip
from PIL import Image
import requests
import os
import sys
import cv2 as cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_attacker import Attacker
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import kornia.filters as kf

"""
DISCLAIMER / WARNING: The following code contains offensive words that are used for research purposes only.
The words used in the tests are used only for research purposes and do not reflect the personal views or beliefs of the authors.
"""

save_path = './output'
if not os.path.exists(save_path):
    os.makedirs(save_path)

device = "cuda" if torch.cuda.is_available() else "cpu"  
model, preprocess = clip.load("ViT-B/32", device=device)  

#we load the normal image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url=url, stream=True).raw)).unsqueeze(0).to(device)

targets = torch.tensor([0]).to(device)
# DISCLAIMER : These offensive words are used for research purposes only, to demonstrate the lack of robustness of the model
classes = ["a car", "a dog", "a cat", "Yassine Guennoun"]

attacker = Attacker(model, classes, device = device, eps=8/255)

adv_image, losses = attacker.attack_specific(image, targets, num_iter=200)

# Save the adversarial image to a file

adv_image_file = os.path.join(save_path, "adversarial_image.png")
save_image(adv_image, adv_image_file)
print(f"Adversarial image saved to {adv_image_file}")

# Plot the loss values and save the plot to a file
plt.figure(figsize=(10, 6))
plt.plot(losses, label="Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss During Adversarial Attack")
plt.legend()

# Save the loss plot
loss_plot_file = os.path.join(save_path, "loss_plot.png")
plt.savefig(loss_plot_file)
print(f"Loss plot saved to {loss_plot_file}")

# finding optimal value of sigma

sigma_values = [x/20 for x in range(1, 100)]
loss_values=[]
result = []
cat_proba = []
car_proba = []

for sig in sigma_values:
    adversarial_image = adv_image.to(device)

    denoised_image = kf.bilateral_blur(adversarial_image, kernel_size=(5, 5), sigma_color=sig, sigma_space=(sig,sig))
    
    print(sig)
    if sig==0.5:
        adv_image_file = os.path.join(save_path, "sig_0.5_image.png")
        save_image(denoised_image, adv_image_file)
    if sig==1:
        adv_image_file = os.path.join(save_path, "sig_1_image.png")
        save_image(denoised_image, adv_image_file)
    if sig==4:
        adv_image_file = os.path.join(save_path, "sig_4_image.png")
        save_image(denoised_image, adv_image_file)

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
sigma_plot_file = os.path.join(save_path, "sigma_plot.png")
plt.savefig(sigma_plot_file)
print(f"Sigma plot saved to {sigma_plot_file}")

# plot the evolution of probability during optimization

plt.figure(figsize=(10, 6))

plt.plot(sigma_values, cat_proba, color='b', label='proba cat')
plt.plot(sigma_values, car_proba, color='r', label='proba car')
plt.xlabel("Sigma")
plt.ylabel("Proba")
plt.title("Proba Depending on Sigma")
plt.legend()

# Save the proba plot
proba_plot_file = os.path.join(save_path, "proba_plot.png")
plt.savefig(proba_plot_file)
print(f"Proba plot saved to {proba_plot_file}")