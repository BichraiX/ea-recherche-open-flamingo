import torch
import clip
from torchvision.datasets import CIFAR100
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
eps = 8/255

# Range of sigma values
sigmas = torch.tensor([i / 5 for i in range(1, 16)])
log_sigmas_100 = torch.logspace(start=torch.log10(torch.tensor(0.01)),
                            end=torch.log10(torch.tensor(1000.0)),
                            steps=40)

unspecific_attack_success_rates = []
denoised_unspecific_attack_success_rates = []
denoised_right_label_rates = []
consistency_success_rates = []
consistency_with_clip = []

def get_one_hot_from_class(class_name):
    return torch.tensor([cifar100.classes.index(class_name)])
# Function to save batch results and return top 5 probabilities
def save_batch_denoised_unspecific_figures(images_and_labels, batch_index, model, attacker, sig):
    batch_ground_truth_preds = []
    batch__denoised_ground_truth_preds = []
    batch_unspecific_attack_preds = []
    batch__denoised_unspecific_attack_preds = []
    labels_list = []
    

    for i, (image, labels) in enumerate(images_and_labels):
        image_input = preprocess(image).unsqueeze(0).to(device)
        model_output = get_one_hot_from_class(attacker.generate_prompt(image_input)).to(device)
        # Perform specific and unspecific attacks
        adv_img_unspecific, _ = attacker.attack_unspecific(image_input, model_output, num_iter=num_iter)

        #denoise the images
        denoised_image_input = kf.bilateral_blur(image_input, kernel_size=(5, 5), sigma_color=sig, sigma_space=(15.8 , 15.8))
        denoised_adv_img_unspecific = kf.bilateral_blur(adv_img_unspecific, kernel_size=(5, 5), sigma_color=sig, sigma_space=(15.8, 15.8))

        # Get predictions
        ground_truth_prediction = attacker.generate_prompt(image_input)
        denoised_ground_truth_prediction = attacker.generate_prompt(denoised_image_input)
        unspecific_attack_prediction = attacker.generate_prompt(adv_img_unspecific.to(device))
        denoised_unspecific_attack_prediction = attacker.generate_prompt(denoised_adv_img_unspecific.to(device))

        # Collect predictions for success calculation later
        batch_ground_truth_preds.append(ground_truth_prediction)
        batch__denoised_ground_truth_preds.append(denoised_ground_truth_prediction)
        batch_unspecific_attack_preds.append(unspecific_attack_prediction)
        batch__denoised_unspecific_attack_preds.append(denoised_unspecific_attack_prediction)
        labels_list.append(cifar100.classes[labels])

    return batch_ground_truth_preds, batch__denoised_ground_truth_preds, batch_unspecific_attack_preds, batch__denoised_unspecific_attack_preds, labels_list

# Create attacker instance
attacker = Attacker(model, cifar100.classes, device=device, eps=eps)

# Loop over different sigma values and calculate success rates
for sig in log_sigmas_100:
    print(f"Processing for sigma = {sig.item()}")
    
    # Reset success counters for this epsilon
    denoised_right_label = 0
    unspecific_attack_success = 0
    denoised_unspecific_attack_success = 0
    normal_predict_success = 0
    same_as_clip = 0

    # Process images in batches
    for batch_index in range(0, n_samples, batch_size):
        batch_images_and_labels = images_and_labels[batch_index:batch_index + batch_size]

        # Save the current batch results as a figure with top 5 predictions
        ground_truth_preds, denoised_ground_truth_preds, unspecific_preds, denoised_unspecific_preds, labels_list = save_batch_denoised_unspecific_figures(
            batch_images_and_labels, batch_index // batch_size, model, attacker, sig.item()
        )

        # Perform success rate calculations using the returned predictions
        for gt_pred, dns_gt_pred, unspec_pred, dns_unspec_pred, label in zip(ground_truth_preds, denoised_ground_truth_preds, unspecific_preds, denoised_unspecific_preds, labels_list):
            if unspec_pred != gt_pred:  # Success if it predicts anything other than the original prediction
                unspecific_attack_success += 1
            if dns_unspec_pred != gt_pred:  # Success if it predicts anything other than the original prediction
                denoised_unspecific_attack_success += 1
            if dns_unspec_pred == label:  # Failure if it gets the right label
                denoised_right_label += 1
            if dns_gt_pred == label:  # Success if it predicts the same as the original label
                normal_predict_success += 1
            if dns_gt_pred == gt_pred: # Success if it predicts the same as the original prediction
                same_as_clip += 1
            

    # Calculate success rates for this sigma
    unspecific_attack_success_rate = unspecific_attack_success / n_samples * 100
    denoised_unspecific_attack_success_rate = denoised_unspecific_attack_success / n_samples * 100
    consistency_success_rate = normal_predict_success / n_samples *100
    denoised_right_label_rate = denoised_right_label / n_samples *100
    same_as_clip_rate = same_as_clip / n_samples *100

    unspecific_attack_success_rates.append(unspecific_attack_success_rate)
    denoised_unspecific_attack_success_rates.append(denoised_unspecific_attack_success_rate)
    denoised_right_label_rates.append(denoised_right_label_rate)
    consistency_success_rates.append(consistency_success_rate)
    consistency_with_clip.append(same_as_clip_rate)

    print(f"Sigma: {sig.item()}, Unspecific Attack Success Rate: {unspecific_attack_success_rate:.2f}%")
    print(f"Sigma: {sig.item()}, Denoised Unspecific Attack Success Rate: {denoised_unspecific_attack_success_rate:.2f}%")
    print(f"Sigma: {sig.item()}, Denoised Unspecific Attack Exact Prediction: {denoised_right_label_rate:.2f}%")
    print(f"Sigma: {sig.item()}, Consistency Success Rate: {consistency_success_rate:.2f}%")
    print(f"Sigma: {sig.item()}, Consistency with CLIP Success Rate: {same_as_clip_rate:.2f}%")

# Plot success rate vs sigma
plt.figure(figsize=(8, 6))
plt.plot(log_sigmas_100.cpu().tolist(), unspecific_attack_success_rates, label="Unspecific Attack", marker='o', color='red')
plt.plot(log_sigmas_100.cpu().tolist(), denoised_unspecific_attack_success_rates, label="Denoised Unspecific Attack", marker='o', color='orange')
plt.plot(log_sigmas_100.cpu().tolist(), denoised_right_label_rates, label="Denoised Right Prediction", marker='o', color='green')
plt.plot(log_sigmas_100.cpu().tolist(), consistency_success_rates, label='Consistency without attack', marker='o', color='blue')
plt.plot(log_sigmas_100.cpu().tolist(), consistency_with_clip, label='Consistency vs CLIP prediction', marker='o', color='purple')
plt.title("Adversarial Unspecific Attack Success Rate vs Sigma")
plt.xlabel("Sigma")
plt.ylabel("Success Rate (%)")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "attack_unspecific_success_vs_sigma.png"))
plt.close()