import torch
import clip
from PIL import Image
import requests
import os
from visual_attacker import Attacker
import matplotlib.pyplot as plt
from torchvision.utils import save_image

"""
DISCLAIMER / WARNING: The following code contains offensive words that are used for research purposes only. (and our fun)
The words used in the tests are used only for research purposes and do not reflect the personal views or beliefs of the authors.
"""

save_path = './output'
if not os.path.exists(save_path):
    os.makedirs(save_path)

device = "cuda" if torch.cuda.is_available() else "cpu"  
model, preprocess = clip.load("ViT-B/32", device=device)  

image = preprocess(Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)).unsqueeze(0).to(device)

targets = torch.tensor([0]).to(device)
# DISCLAIMER : These offensive words are used for research purposes only, to demonstrate the lack of robustness of the model
classes = ["a nigger", "a dog", "a cat", "Yassine Guennoun"]

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

