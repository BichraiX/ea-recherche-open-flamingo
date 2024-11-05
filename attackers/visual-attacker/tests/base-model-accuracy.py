import torch
import clip
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
import os

# Load CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load CIFAR-100 dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)

# Prepare the text inputs (class names of CIFAR-100)
class_names = cifar100.classes
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)

# Function to calculate accuracy
def calculate_clip_accuracy(model, dataset, text_inputs, num_images=100):
    correct_predictions = 0
    
    for i in range(num_images):
        image, label = dataset[i]  # Get image and ground truth label
        image_input = image.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            # Get image and text features
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity (dot product) and choose the class with highest similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            prediction = similarity.argmax(dim=-1).item()

        # Check if the prediction matches the ground truth
        if prediction == label:
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / num_images * 100
    return accuracy

# Calculate accuracy on the first 100 images
accuracy = 0
for _ in range(10):
    accuracy += calculate_clip_accuracy(model, cifar100, text_inputs, num_images=100)
print(f"CLIP accuracy on the first 100 images of CIFAR-100: {accuracy/10:.2f}%")
