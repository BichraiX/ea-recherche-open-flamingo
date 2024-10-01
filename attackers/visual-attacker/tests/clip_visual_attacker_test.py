import torch
import clip
from PIL import Image
import requests
import os
from visual_attacker import Attacker
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Ensure the necessary directories exist
save_path = './output'
if not os.path.exists(save_path):
    os.makedirs(save_path)

device = "cuda" if torch.cuda.is_available() else "cpu"  # Store data on GPU if available
model, preprocess = clip.load("ViT-B/32", device=device)  # Load the model and preprocessing function

# Download and preprocess the image
image = preprocess(Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)).unsqueeze(0).to(device)

# Define the target and class labels
targets = torch.tensor([0]).to(device)
classes = ["a nigger", "a dog", "a cat", "Yassine Guennoun"]  # Your class names

# Initialize the Attacker
attacker = Attacker(model, targets, classes, eps=8/255)

# Perform the adversarial attack
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

# Optionally show the plot (if you want to visualize it during the execution)
plt.show()
