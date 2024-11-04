import torch
import clip
from PIL import Image
from torchvision.datasets import CIFAR100
import os
import sys

import optuna
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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


def get_one_hot_from_class(class_name):
    return torch.tensor([cifar100.classes.index(class_name)])
# Function to save batch results and return top 5 probabilities
def save_batch_denoised_specific_figures(images_and_labels, batch_index, model, attacker, sig_color, sig_space, kernel):
    batch__denoised_specific_attack_preds = []
    labels_list = []

    for i, (image, labels) in enumerate(images_and_labels):
        image_input = preprocess(image).unsqueeze(0).to(device)
        target = torch.tensor([target_class_id]).to(device)  # Set target class for specific attack
        # Perform specific and unspecific attacks
        adv_img_specific, _ = attacker.attack_specific(image_input, target, num_iter=num_iter)

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
        denoised_specific_preds, labels_list = save_batch_denoised_specific_figures(
            batch_images_and_labels, batch_index // batch_size, model, attacker, sig_color, sig_space, kernel
        )

        # Perform success rate calculations using the returned predictions
        for dns_spec_pred, label in zip(denoised_specific_preds, labels_list):
            if dns_spec_pred == label:  # Success if it gets the right label
                denoised_right_label += 1

    # Calculate success rates for this sigma
    denoised_right_label_rate = denoised_right_label / n_samples *100
    return((-1)*denoised_right_label_rate)

def objective(trial):
    # Log-uniforme pour obtenir une distribution exponentielle
    param1 = trial.suggest_uniform('param1', 0, 3)  # Distribution exponentielle entre 0 et 100
    param2 = trial.suggest_uniform('param2', 0, 80) # Distribution uniforme
    param3 = trial.suggest_categorical('param3', list(range(7, 8, 2)))
    
    # Calcul du score avec la fonction d'évaluation
    score = evaluate_specific(param1, param2, param3)
    return score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

best_params = study.best_params
best_score = study.best_value
print("Meilleurs paramètres :", best_params)
print("Meilleur score :", best_score)

# Récupération des valeurs des essais pour visualisation
trials_df = study.trials_dataframe()

# Visualisation 3D des résultats
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Récupération des valeurs depuis le DataFrame d'Optuna
x = torch.tensor(trials_df['params_param1'].values, dtype=torch.float32)
y = torch.tensor(trials_df['params_param2'].values, dtype=torch.float32)
z = torch.tensor(trials_df['params_param3'].values, dtype=torch.float32)
scores = torch.tensor(trials_df['value'].values, dtype=torch.float32)

# Création d'un nuage de points avec couleur en fonction du score
scatter = ax.scatter(x.cpu(), y.cpu(), z.cpu(), c=scores.cpu(), cmap='viridis', s=50)
fig.colorbar(scatter, ax=ax, label="Score")
ax.set_xlabel('Paramètre 1')
ax.set_ylabel('Paramètre 2')
ax.set_zlabel('Paramètre 3')
plt.title("Optimisation des hyperparamètres et scores (échelle exponentielle)")

# Affichage du graphique 3D
plt.savefig(os.path.join(output_dir, "zoom_hyperparams_3D_optimization_specific.png"))
plt.close()

# Visualisation en 2D (contour plot) pour les paramètres 1 et 2
plt.figure(figsize=(10, 7))
plt.tricontourf(x.cpu(), y.cpu(), scores.cpu(), levels=14, cmap="viridis")
plt.colorbar(label="Score")
plt.scatter(best_params['param1'], best_params['param2'], color="red", label="Meilleur")
plt.xlabel("Paramètre 1")
plt.ylabel("Paramètre 2")
plt.title("Contour des scores pour les paramètres 1 et 2")
plt.legend()

plt.savefig(os.path.join(output_dir, "zoom_hyperparams_contour_2D_optimization_specific.png"))
plt.close()