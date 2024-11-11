import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToPILImage
from visual_attacker import Attacker
from defense import RandomizedSmoothing
import time

# Create directory to save the results
save_path = './output'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Download and load the CIFAR-100 test set
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
n_samples = 1
fixed_indices = list(range(n_samples)) 
images_and_labels = [(cifar100[i][0], cifar100[i][1]) for i in fixed_indices]

# Define sigma values for Randomized Smoothing
#sigma_values = [0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,1]
#sigma_values = [0.01,0.05,0.1,0.15,0.2]
sigma_values = [0]
def evaluate_with_attack_time(model, images_and_labels, cifar100, device, n_samples, attack_function, sigma_values):
    accuracy_results = []
    execution_times = []
    
    # Loop over each sigma value
    for sigma in sigma_values:
        times = []
        correct_predictions = 0  # Counter for correct predictions
        clip_count = 0
        # Start measuring the time for this sigma
        for image, label in images_and_labels:
            # Preprocess the image
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            target = torch.tensor([0]).to(device)
            
            
            # Run the provided attack function
            attacker = Attacker(model, cifar100.classes, device=device, eps=0/255)
            #adv_image, _ = attack_function(attacker,image_input,target,num_iter=11)
            res= attacker.generate_prompt(image_input)
            
            if res == cifar100.classes[label] :
                clip_count+=1

            start_time = time.time()
            # Defense using Randomized Smoothing
            defense = RandomizedSmoothing(model, cifar100.classes, num_samples=100, sigma=sigma, preprocess = preprocess)
    
            result = defense.predict(image_input, False)

            prediction, _ = result
            
            print(cifar100.classes[prediction])
            print(cifar100.classes[label])
            if prediction == label:
                correct_predictions += 1
            times.append(time.time()-start_time)

        # Calculate accuracy for the current sigma
        accuracy = correct_predictions / n_samples * 100
        accuracy_results.append(accuracy)

        # Stop measuring the time and calculate elapsed time
        elapsed_time = np.mean(np.array(times))
        execution_times.append(elapsed_time)
        print ( f"CLIP accuracy :{clip_count: .2f}")
        print(f"Sigma: {sigma}, Accuracy: {accuracy:.2f}%, Time: {elapsed_time:.2f}s")



    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot accuracy on the first Y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Sigma values')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(sigma_values, accuracy_results, marker='o', color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second Y-axis for execution time
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Time (s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(sigma_values, execution_times, marker='x', color=color, label='Execution Time')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a title
    plt.title("Accuracy and Execution Time vs Sigma")

    # Save or show the plot
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "accuracy_and_time_vs_sigma.png"))
        print(f"Plot saved at {os.path.join(save_path, 'accuracy_and_time_vs_sigma.png')}")
    else:
        plt.show()

    plt.close()
evaluate_with_attack_time(model, images_and_labels, cifar100, device, n_samples, Attacker.attack_specific, sigma_values)


