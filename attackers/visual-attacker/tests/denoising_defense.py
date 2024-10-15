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

attacker = Attacker(model, targets, classes, eps=8/255)

adv_image, losses = attacker.attack(image, num_iter=200)

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

# finding optimal value of d and sigma

sigma_values = [2*x for x in range(1, 50)]
loss_values=[]
result = []
cat_proba = []
car_proba = []

for sig in sigma_values:
    adversarial_image = adv_image
    if adversarial_image.is_cuda:
        adversarial_image = adversarial_image.cpu()

    adversarial_image_np = adversarial_image.detach().numpy()
    adversarial_image_np = adversarial_image_np.transpose(1, 2, 0)  # (C, H, W) to (H, W, C) for RGB
    adversarial_image_np = np.clip(adversarial_image_np * 255, 0, 255).astype(np.uint8)

    denoised_image = cv2.bilateralFilter(adversarial_image_np, d=-1, sigmaColor=sig, sigmaSpace=sig)
    rgb_denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    final_denoised_image = Image.fromarray(rgb_denoised_image)

    denoised_image_tensor = preprocess(final_denoised_image).unsqueeze(0).to(device)

    print(sig)
    if sig==10:
        adv_image_file = os.path.join(save_path, "sig_10_image.png")
        save_image(denoised_image_tensor, adv_image_file)
    if sig==30:
        adv_image_file = os.path.join(save_path, "sig_30_image.png")
        save_image(denoised_image_tensor, adv_image_file)
    if sig==90:
        adv_image_file = os.path.join(save_path, "sig_90_image.png")
        save_image(denoised_image_tensor, adv_image_file)

    loss_values.append(attacker.loss(denoised_image_tensor).item())

    result.append(attacker.predict(denoised_image_tensor))
    cat_proba.append(attacker.proba_vect(denoised_image_tensor)[0, 2].item())
    car_proba.append(attacker.proba_vect(denoised_image_tensor)[0, 0].item())

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