import os
import clip
import torch
import random
import numpy as np
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visual_attacker import Attacker

# Create directory to save the results
output_dir = "attack_results"
os.makedirs(output_dir, exist_ok=True)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
num_iter = 11

# Download the CIFAR-100 dataset and pick consistent images
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
n_samples = 100 
fixed_indices = list(range(n_samples)) 
images_and_labels = [(cifar100[i][0], cifar100[i][1]) for i in fixed_indices]
target_class_id = 0
batch_size = 2

# Range of epsilon values (1/255 to 10/255)
epsilons = [i / 255 for i in range(1, 11)]
specific_attack_success_rates = []
unspecific_attack_success_rates = []

def get_one_hot_from_class(class_name):
    return torch.tensor([cifar100.classes.index(class_name)])
# Function to save batch results and return top 5 probabilities
def save_batch_figures(images_and_labels, batch_index, model, attacker):
    fig, axes = plt.subplots(len(images_and_labels), 2, figsize=(10, len(images_and_labels) * 5))

    batch_ground_truth_preds = []
    batch_specific_attack_preds = []
    batch_unspecific_attack_preds = []

    for i, (image, class_id) in enumerate(images_and_labels):
        image_input = preprocess(image).unsqueeze(0).to(device)
        target = torch.tensor([target_class_id]).to(device)  # Set target class for specific attack
        model_output = get_one_hot_from_class(attacker.generate_prompt(image_input)).to(device)
        # Perform specific and unspecific attacks
        adv_img_specific, _ = attacker.eot_attack_specific(image_input, target, num_iter=num_iter)
        adv_img_unspecific, _ = attacker.eot_attack_unspecific(image_input, model_output, num_iter=num_iter)

        # Get predictions
        ground_truth_prediction = attacker.generate_prompt(image_input)
        specific_attack_prediction = attacker.generate_prompt(adv_img_specific.to(device))
        unspecific_attack_prediction = attacker.generate_prompt(adv_img_unspecific.to(device))

        # Collect predictions for success calculation later
        batch_ground_truth_preds.append(ground_truth_prediction)
        batch_specific_attack_preds.append(specific_attack_prediction)
        batch_unspecific_attack_preds.append(unspecific_attack_prediction)

    return batch_ground_truth_preds, batch_specific_attack_preds, batch_unspecific_attack_preds

# Loop over different epsilon values and calculate success rates
for eps in epsilons:
    print(f"Processing for epsilon = {eps}")
    
    # Reset success counters for this epsilon
    specific_attack_success = 0
    unspecific_attack_success = 0

    # Process images in batches
    for batch_index in range(0, n_samples, batch_size):
        batch_images_and_labels = images_and_labels[batch_index:batch_index + batch_size]

        # Create attacker instance for this batch
        attacker = Attacker(model, cifar100.classes, device=device, eps=eps)

        # Save the current batch results as a figure with top 5 predictions
        ground_truth_preds, specific_preds, unspecific_preds = save_batch_figures(
            batch_images_and_labels, batch_index // batch_size, model, attacker
        )

        # Perform success rate calculations using the returned predictions
        for gt_pred, spec_pred, unspec_pred in zip(ground_truth_preds, specific_preds, unspecific_preds):
            if spec_pred == cifar100.classes[target_class_id]:  # Success if it predicts the target
                specific_attack_success += 1
            if unspec_pred != gt_pred:  # Success if it predicts anything other than the original prediction
                unspecific_attack_success += 1

    # Calculate success rates for this epsilon
    specific_attack_success_rate = specific_attack_success / n_samples * 100
    unspecific_attack_success_rate = unspecific_attack_success / n_samples * 100

    specific_attack_success_rates.append(specific_attack_success_rate)
    unspecific_attack_success_rates.append(unspecific_attack_success_rate)

    print(f"Epsilon: {eps}, EOT Specific Attack Success Rate: {specific_attack_success_rate:.2f}%")
    print(f"Epsilon: {eps}, EOT Unspecific Attack Success Rate: {unspecific_attack_success_rate:.2f}%")

# Plot success rate vs epsilon
plt.figure(figsize=(8, 6))
plt.plot(epsilons, specific_attack_success_rates, label="EOT Specific Attack", marker='o', color='blue')
plt.plot(epsilons, unspecific_attack_success_rates, label="EOT Unspecific Attack", marker='o', color='red')
plt.title("Adversarial Attack Success Rate vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Success Rate (%)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "eot_attack_success_vs_epsilon.png"))
plt.close()

