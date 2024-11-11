import torch
import clip
from PIL import Image
from torchvision.datasets import CIFAR100
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_attacker import Attacker
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import kornia.filters as kf
import seaborn as sns

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
eps = 8/255
kernel=7

# Range of sigma values
sigmas = torch.tensor([i for i in range(1, 51)])
log_sigmas_10 = torch.logspace(start=torch.log10(torch.tensor(0.01)),
                            end=torch.log10(torch.tensor(10.0)),
                            steps=15)
log_sigmas_100 = torch.logspace(start=torch.log10(torch.tensor(0.05)),
                            end=torch.log10(torch.tensor(500.0)),
                            steps=25)

denoised_right_label_rates = []


def get_one_hot_from_class(class_name):
    return torch.tensor([cifar100.classes.index(class_name)])
# Function to save batch results and return top 5 probabilities
def save_batch_denoised_unspecific_figures(images_and_labels, batch_index, model, attacker, sig_color, sig_space, kernel):
    batch__denoised_specific_attack_preds = []
    labels_list = []

    for i, (image, labels) in enumerate(images_and_labels):
        image_input = preprocess(image).unsqueeze(0).to(device)
        model_output = get_one_hot_from_class(attacker.generate_prompt(image_input)).to(device)
        # Perform specific and unspecific attacks
        adv_img_specific, _ = attacker.attack_unspecific(image_input, model_output, num_iter=num_iter)

        #denoise the images
        denoised_adv_img_specific = kf.bilateral_blur(adv_img_specific, kernel_size=(kernel, kernel), sigma_color=sig_color, sigma_space=(sig_space,sig_space))

        # Get predictions
        denoised_specific_attack_prediction = attacker.generate_prompt(denoised_adv_img_specific.to(device))

        # Collect predictions for success calculation later
        batch__denoised_specific_attack_preds.append(denoised_specific_attack_prediction)
        labels_list.append(cifar100.classes[labels])

    return batch__denoised_specific_attack_preds, labels_list

# Create attacker instance
attacker = Attacker(model, cifar100.classes, device=device, eps=eps)

# Loop over different sigma values and calculate -1*success rates
def evaluate_specific(sig_color, sig_space, kernel=kernel):
    # Reset success counters for this epsilon
    denoised_right_label = 0

    # Process images in batches
    for batch_index in range(0, n_samples, batch_size):
        batch_images_and_labels = images_and_labels[batch_index:batch_index + batch_size]

        # Save the current batch results as a figure with top 5 predictions
        denoised_specific_preds, labels_list = save_batch_denoised_unspecific_figures(
            batch_images_and_labels, batch_index // batch_size, model, attacker, sig_color, sig_space, kernel
        )

        # Perform success rate calculations using the returned predictions
        for dns_spec_pred, label in zip(denoised_specific_preds, labels_list):
            if dns_spec_pred == label:  # Success if it gets the right label
                denoised_right_label += 1

    # Calculate success rates for this sigma
    denoised_right_label_rate = denoised_right_label / n_samples *100
    return((-1)*denoised_right_label_rate)

scores = []
X= log_sigmas_100
Y= log_sigmas_100
x_mini, y_mini = 0,0
mini = 0

for y in Y:
    l=[]
    for x in X:
        score = evaluate_specific(x.item(), y.item(), kernel=kernel)
        l.append(score)
        if score<mini:
            mini = score
            x_mini, y_mini = x.item(),y.item()
    scores.append(l)

print(mini)
print(x_mini)
print(y_mini)

scores_tensor = torch.tensor(scores)


plt.figure(figsize=(8, 6))
sns.heatmap(scores_tensor.cpu().numpy(), annot=True, cmap="viridis", xticklabels=[f"{x:.2f}" for x in X.cpu().tolist()], yticklabels=[f"{x:.2f}" for x in Y.cpu().tolist()])

plt.xlabel('sigma_color')
plt.ylabel('sigma_space')
plt.title('Heatmap of Sigma Optimization for Unspecific Attack')

plt.savefig(os.path.join(output_dir, "hyperparams_optimization_unspecific.png"))
plt.close()