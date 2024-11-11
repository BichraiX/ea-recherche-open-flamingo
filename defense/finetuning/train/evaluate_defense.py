import sys
import os

import torch
import torchvision.transforms as transforms
import open_clip
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../test_defense_denoising'))

from visual_attacker import Attacker

import json
import time
import numpy as np

import torch.nn.functional as F
from torchvision.transforms import Resize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CLIP_eval.eval_utils import load_clip_model  # Import load function
import argparse
from train.utils import str2bool

file_to_save = "evaluation_ft"
os.makedirs(file_to_save, exist_ok=True)

# Set up arguments to specify the model type, pretrained weights, and fine-tuned weights file
parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_name', type=str, default='ViT-B-32', help='Model name, e.g., ViT-L-14 or ViT-B-32')
parser.add_argument('--pretrained', type=str, default='openai', help='Specify "openai" for pretrained weights')
parser.add_argument('--finetuned_weights', type=str, default='output_dir/ViT-L-14_openai_cifar100_l2_cifar100_testcifar100/checkpoints/final.pt', help='Path to the fine-tuned weights file')
parser.add_argument('--samples', type=int, default=100, help='The number of images to evaluate on')
parser.add_argument('--output_normalize', type=str2bool, default=False, help='Whether the embedding is normalized')
parser.add_argument('--beta', type=float, default=0., help='Model interpolation parameter')
parser.add_argument('--eps', type=float, default=4/255, help='epsilon of attack')
args = parser.parse_args()

tokenizer = open_clip.get_tokenizer(args.clip_model_name)

# Load the model architecture
model_orig,_,preprocess = open_clip.create_model_and_transforms(args.clip_model_name, pretrained='laion2b_s34b_b79k')
#model_orig,_,_ = load_clip_model(args.clip_model_name, pretrained='openai', beta=args.beta)

model, _, image_processor = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
#model, preprocessor_without_normalize, normalize = load_clip_model(args.clip_model_name, args.pretrained, beta=args.beta)

# Load the fine-tuned weights from final.pt
checkpoint = torch.load(args.finetuned_weights)

model.load_state_dict(checkpoint, strict=False)
print(f"Loaded fine-tuned weights from {args.finetuned_weights}")
#model = torch.load(args.finetuned_weights)


# Load CIFAR-100 dataset
preprocessor_without_normalize = transforms.Compose(preprocess.transforms[:-1])
normalize = preprocess.transforms[-1]
transform = preprocessor_without_normalize
dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)
target_class_id = 0
batch_size = 2
eps = args.eps
num_iter = 11

cifar100_classes = dataset.classes

subset = Subset(dataset, list(range(args.samples)))  # Use the first 100 images
data_loader = DataLoader(subset, batch_size=10, shuffle=False)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_device = 0
model.eval()  # Set model to evaluation mode
model.to(device)
model_orig.eval() # Set model to evaluation mode
model_orig.to(device)

model.eval()  # Set model to evaluation mode

text_inputs = open_clip.tokenize(cifar100_classes).to(device)

attacker = Attacker(model, cifar100_classes, device=device, eps=eps)

# Evaluate the model
correct_orig = 0
same_as_clip = 0
correct_ft = 0
correct_orig_adv = 0
correct_ft_adv = 0
same_as_clip_adv = 0
total = 0

for images, labels in data_loader:
    images, labels = images.to(device), labels.to(device)

    target = torch.tensor([target_class_id]*10).to(device)  # Set target class for specific attack
    # Perform specific and unspecific attacks
    adv_img_specific, _ = attacker.attack_specific_v2(images, target, num_iter=num_iter)
    adv_img_specific.to(device)
    model_orig.to(device)

    with torch.no_grad():
        adv_img_specific = adv_img_specific.to(next(model.parameters()).device)

        outputs_orig = model_orig(images, text_inputs)
        outputs_orig_adv = model_orig(adv_img_specific, text_inputs)

        outputs = model(images, text_inputs)
        outputs_adv = model(adv_img_specific, text_inputs)

        image_features_orig, text_features_orig, _ = outputs_orig
        image_features_orig_adv, text_features_orig_adv, _ = outputs_orig_adv
        image_features, text_features, _ = outputs
        image_features_adv, text_features_adv, _ = outputs_adv

        similarity_orig = image_features_orig @ text_features_orig.T
        _, predicted_orig = torch.max(similarity_orig, 1)

        similarity_orig_adv = image_features_orig_adv @ text_features_orig_adv.T
        _, predicted_orig_adv = torch.max(similarity_orig_adv, 1)

        similarity = image_features @ text_features.T
        _, predicted = torch.max(similarity, 1)

        similarity_adv = image_features_adv @ text_features_adv.T
        _, predicted_adv = torch.max(similarity_adv, 1)

        total += labels.size(0)
        correct_orig += (predicted_orig == labels).sum().item()
        same_as_clip += (predicted_orig == predicted).sum().item()
        correct_ft += (predicted == labels).sum().item()
        correct_orig_adv += (predicted_orig_adv == labels).sum().item()
        correct_ft_adv += (predicted_adv == labels).sum().item()
        same_as_clip_adv += (predicted_adv == predicted_orig_adv).sum().item()

accuracy_orig = 100 * correct_orig / total
print(f'Accuracy on the first 100 images of CIFAR-100 with original model: {accuracy_orig:.2f}%')
consistency_vs_clip = 100 * same_as_clip / total
print(f'Consistency vs original model on the first 100 images of CIFAR-100 with finetuned model: {consistency_vs_clip:.2f}%')
accuracy_ft = 100 * correct_ft / total
print(f'Accuracy on the first 100 images of CIFAR-100 with finetuned model: {accuracy_ft:.2f}%')
accuracy_orig_adv = 100 * correct_orig_adv / total
print(f'Accuracy on the first 100 attacked images of CIFAR-100 with original model: {accuracy_orig_adv:.2f}%')
accuracy_ft_adv = 100 * correct_ft_adv / total
print(f'Accuracy on the first 100 attacked images of CIFAR-100 with finetuned model: {accuracy_ft_adv:.2f}%')
consistency_vs_clip_adv = 100 * same_as_clip_adv / total
print(f'Consistency vs original model on the first 100 attacked images of CIFAR-100 with finetuned model: {consistency_vs_clip_adv:.2f}%')

accuracies = [accuracy_orig, accuracy_orig_adv, accuracy_ft, accuracy_ft_adv]

title = [
    "Original Model\nNormal Images",
    "Original Model\nAttacked Images",
    "Fine-tuned Model\nNormal Images",
    "Fine-tuned Model\nAttacked Images"
]

x = [0,1,2,3]

plt.bar(x, accuracies, color=['skyblue', 'salmon', 'lightgreen', 'lightcoral'], width=0.6)

# Add text annotations on top of each bar
for i, acc in enumerate(accuracies):
    plt.text(x[i], acc + 0.02, f'{acc:.2f}', ha='center', va='bottom')

plt.xlabel("Model and Image Type")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison on 100 first CIFAR-100 Images")

plt.xticks(x, title)
plt.tight_layout()

plt.grid(True)
plt.savefig(os.path.join(file_to_save, "accuracy_models_attacked.png"))
plt.close()