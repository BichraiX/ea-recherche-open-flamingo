import torch
import clip
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torchvision.utils import save_image
from visual_attacker import Attacker
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

# Create output directory if it doesn't exist
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Settings
num_iter = 11
n_images = 20  # Number of images to attack (first 20)
eps = 8 / 255  # Perturbation for adversarial attacks
images_per_file = 4  # Number of images per file

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocess function
model, preprocess = clip.load("ViT-B/32", device=device)

# Load CIFAR-100 dataset and get the first 20 images
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
classes = cifar100.classes
images_and_labels = [(cifar100[i][0], cifar100[i][1]) for i in range(n_images)]

# Initialize the attacker
attacker = Attacker(model, classes, device=device, eps=eps)

# Initialize OpenFlamingo model (ensure that it's installed and configured)
flamingo_model, flamingo_image_processor, flamingo_tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
flamingo_model.load_state_dict(torch.load(checkpoint_path), strict=False)

# Process the images in batches of 4
for batch_idx in range(0, n_images, images_per_file):
    fig, axes = plt.subplots(images_per_file, 2, figsize=(12, 5 * images_per_file))
    for i in range(images_per_file):
        image_idx = batch_idx + i
        if image_idx >= n_images:
            break

        image, label = images_and_labels[image_idx]
        print(f"\nProcessing image {image_idx + 1}/{n_images}...")

        # Preprocess the image
        image_input = preprocess(image).unsqueeze(0).to(device)
        target = torch.tensor([label]).to(device)

        # Perform specific attack
        adv_image_specific, losses_specific = attacker.attack_specific(image_input, target, num_iter=num_iter)

        # Perform unspecific attack (using the model's original output as the target to push away from)
        model_output = torch.tensor([cifar100.classes.index(attacker.generate_prompt(image_input))]).to(device)
        adv_image_unspecific, losses_unspecific = attacker.attack_unspecific(image_input, model_output, num_iter=num_iter)

        # Prepare the adversarial image for OpenFlamingo model input
        adv_image_specific_for_flamingo = adv_image_specific.unsqueeze(1).unsqueeze(0).to(device)
        adv_image_unspecific_for_flamingo = adv_image_unspecific.unsqueeze(1).unsqueeze(0).to(device)
        flamingo_tokenizer.padding_side = "left"
        lang_x = flamingo_tokenizer(
            ["<image>An image of"],
            return_tensors="pt",
        )
        
        # Pass the adversarial images through OpenFlamingo
        with torch.no_grad():
            flamingo_specific_output = flamingo_model.generate(vision_x=adv_image_specific_for_flamingo.cpu(), lang_x=lang_x["input_ids"], attention_mask=lang_x["attention_mask"], max_new_tokens=20, num_beams=3)
            flamingo_unspecific_output = flamingo_model.generate(vision_x=adv_image_unspecific_for_flamingo.cpu(), lang_x=lang_x["input_ids"], attention_mask=lang_x["attention_mask"], max_new_tokens=20, num_beams=3)

        # Decode OpenFlamingo outputs
        specific_output_text = flamingo_tokenizer.decode(flamingo_specific_output[0])
        unspecific_output_text = flamingo_tokenizer.decode(flamingo_unspecific_output[0])

        # Plot the original image on the left
        axes[i, 0].imshow(image)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Original Image (Label: {classes[label]})")

        # Plot the Flamingo outputs for the specific and unspecific attacks on the right
        text_display = f"Specific Attack Output:\n{specific_output_text}\n\nUnspecific Attack Output:\n{unspecific_output_text}"
        axes[i, 1].text(0.1, 0.5, text_display, fontsize=12, va="center", ha="left", wrap=True)
        axes[i, 1].axis('off')

    # Save the figure with 4 images and their outputs
    fig.suptitle(f"Adversarial Attacks and OpenFlamingo Outputs (Images {batch_idx+1} to {min(batch_idx+4, n_images)})")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Add space for the main title
    plt.savefig(os.path.join(output_dir, f"adversarial_attacks_{batch_idx+1}_to_{min(batch_idx+4, n_images)}.png"))
    plt.close()

print("Adversarial attacks and OpenFlamingo outputs processing completed.")
