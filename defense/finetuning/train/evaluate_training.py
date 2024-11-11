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

checkpoints_model = []
checkpoints_model.append("model_orig")
checkpoints_model.append("output_dir/ViT-B-32_openai_cifar100_ce_cifar100_testcifar100/checkpoints/step_1000_opt.pt")
checkpoints_model.append("output_dir/ViT-B-32_openai_cifar100_ce_cifar100_testcifar100/checkpoints/step_2000_opt.pt")
checkpoints_model.append("output_dir/ViT-B-32_openai_cifar100_ce_cifar100_testcifar100/checkpoints/step_3000_opt.pt")
checkpoints_model.append("output_dir/ViT-B-32_openai_cifar100_ce_cifar100_testcifar100/checkpoints/step_4000_opt.pt")
checkpoints_model.append("output_dir/ViT-B-32_openai_cifar100_ce_cifar100_testcifar100/checkpoints/step_5000_opt.pt")
checkpoints_model.append("output_dir/ViT-B-32_openai_cifar100_ce_cifar100_testcifar100/checkpoints/step_6000_opt.pt")
checkpoints_model.append("output_dir/ViT-B-32_openai_cifar100_ce_cifar100_testcifar100/checkpoints/step_7000_opt.pt")
checkpoints_model.append("output_dir/ViT-B-32_openai_cifar100_ce_cifar100_testcifar100/checkpoints/step_8000_opt.pt")
checkpoints_model.append("output_dir/ViT-B-32_openai_cifar100_ce_cifar100_testcifar100/checkpoints/step_9000_opt.pt")


# Set up arguments to specify the model type, pretrained weights, and fine-tuned weights file
parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_name', type=str, default='ViT-B-32', help='Model name, e.g., ViT-L-14 or ViT-B-32')
parser.add_argument('--pretrained', type=str, default='openai', help='Specify "openai" for pretrained weights')
parser.add_argument('--samples', type=int, default=100, help='The number of images to evaluate on')
parser.add_argument('--output_normalize', type=str2bool, default=False, help='Whether the embedding is normalized')
parser.add_argument('--beta', type=float, default=0., help='Model interpolation parameter')
args = parser.parse_args()

tokenizer = open_clip.get_tokenizer(args.clip_model_name)

# Load the model architecture
model_orig,_,preprocess = open_clip.create_model_and_transforms(args.clip_model_name, pretrained='laion2b_s34b_b79k')
#model_orig,_,_ = load_clip_model(args.clip_model_name, pretrained='openai', beta=args.beta)


# Load CIFAR-100 dataset
preprocessor_without_normalize = transforms.Compose(preprocess.transforms[:-1])
normalize = preprocess.transforms[-1]
transform = preprocessor_without_normalize
dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)
target_class_id = 0
batch_size = 2
eps = 4/255
num_iter = 11

cifar100_classes = dataset.classes

subset = Subset(dataset, list(range(args.samples)))  # Use the first 100 images
data_loader = DataLoader(subset, batch_size=10, shuffle=False)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_device = 0

text_inputs = open_clip.tokenize(cifar100_classes).to(device)

accuracies_final = []
accuracies_adv_final = []

for cck in checkpoints_model:

    if cck == "model_orig":
        model = model_orig
        print(f"Loaded fine-tuned weights from {cck}")
        model.eval() # Set model to evaluation mode
        model.to(device)
        model.eval()  # Set model to evaluation mode
        
        attacker = Attacker(model, cifar100_classes, device=device, eps=eps)

    else:
        model, _,_ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        #model, preprocessor_without_normalize, normalize = load_clip_model(args.clip_model_name, args.pretrained, beta=args.beta)

        # Load the fine-tuned weights from final.pt
        checkpoint = torch.load(cck)

        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded fine-tuned weights from {cck}")
        #model = torch.load(args.finetuned_weights)

        attacker = Attacker(model, cifar100_classes, device=device, eps=eps)

        model.to(device)
        model.eval()  # Set model to evaluation mode

    # Evaluate the model
    
    correct_ft = 0
    correct_ft_adv = 0
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


            outputs = model(images, text_inputs)
            outputs_adv = model(adv_img_specific, text_inputs)

            image_features, text_features, _ = outputs
            image_features_adv, text_features_adv, _ = outputs_adv


            similarity = image_features @ text_features.T
            _, predicted = torch.max(similarity, 1)

            similarity_adv = image_features_adv @ text_features_adv.T
            _, predicted_adv = torch.max(similarity_adv, 1)

            total += labels.size(0)
            correct_ft += (predicted == labels).sum().item()
            correct_ft_adv += (predicted_adv == labels).sum().item()

    accuracy_ft = 100 * correct_ft / total
    print(f'Accuracy on the first 100 images of CIFAR-100 with finetuned model: {accuracy_ft:.2f}%')
    accuracy_ft_adv = 100 * correct_ft_adv / total
    print(f'Accuracy on the first 100 attacked images of CIFAR-100 with finetuned model: {accuracy_ft_adv:.2f}%')
    accuracies_final.append(accuracy_ft)
    accuracies_adv_final.append(accuracy_ft_adv)

absc = [1000*x for x in range(len(accuracies_final))]

plt.figure(figsize=(8, 6))
plt.plot(absc, accuracies_final, label="Accuracy on normal images", marker='o', color='blue')
plt.plot(absc, accuracies_adv_final, label="Accuracy on attacked images", marker='o', color='red')

plt.title("Evolution of accuracy during finetuning")
plt.xlabel("Steps")
plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)
plt.savefig(os.path.join(file_to_save, "accuracy_model_evolution.png"))
plt.close()