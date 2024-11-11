import torch
import clip
import numpy as np
from tqdm import tqdm
import scipy.stats
from statsmodels.stats.proportion import proportion_confint
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import os

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images
    


class RandomizedSmoothing:
    def __init__(self, model, classes, num_samples, sigma,preprocess):
        self.model = model
        self.preprocess= preprocess
        self.classes = classes
        self.num_samples = num_samples
        self.sigma = sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text = clip.tokenize(self.classes).to(self.device)  # Initialisation des classes tokenisées
        self.model.eval()
        self.model.requires_grad_(False)


    def sample_under_noise(self, x):
        counts = np.zeros(len(self.classes))
        x=x.to(self.device)
        
        for _ in range(self.num_samples):
            noise = (torch.randn_like(x) * self.sigma).to(self.device)
        
            #x = denormalize(x).clone().to(self.device)
            #noise.data = (noise.data + x.data).clamp(0, 1) - x.data
            #noisy_image= normalize(x + noise)

            #noise = (torch.randn_like(x) * self.sigma).to(self.device)
            #noisy_image = (x + noise).to(self.device)
            #noisy_image = torch.clamp(noisy_image, 0, 1)
            #noisy_image = normalize(noisy_image)
            noisy_image = x + noise
            to_pil = ToPILImage()



            with torch.no_grad():
                processed_image = self.preprocess(to_pil(noisy_image[0])).unsqueeze(0).to(self.device)
            
                logits_per_image, _ = self.model(processed_image, self.text)  
                
            probs = torch.nn.functional.softmax(logits_per_image, dim=-1)
            
            pred = probs.argmax().item()
            counts[pred] += 1
        return counts

    @staticmethod
    def binom_p_value(nA, nA_plus_nB, p):
        """Return the p-value of the two-sided hypothesis test."""
        return scipy.stats.binomtest(nA,nA_plus_nB,p)

    def lower_conf_bound(self, k, n, alpha):
        """Return a one-sided lower confidence interval."""
        return proportion_confint(k, n, alpha=2*alpha, method="beta")[0]

    def predict(self, x, test,alpha=0.05):
        """Make a prediction with randomized smoothing."""
        counts = self.sample_under_noise(x)
        nA = counts.max()
        nB = counts[counts.argsort()[-2]]  # Second largest count

        # Calculate p-value and decide whether to abstain or predict
        p_value = self.binom_p_value(int(nA), int(nA + nB), 0.5).pvalue
        if test :
            if p_value < alpha:
                
                predicted_class = counts.argmax()

                # Calculate certified radius
                pA = nA / (nA+nB)
                R = self.sigma * (np.sqrt(2) * (np.sqrt(nA + nB) * np.abs(pA - 0.5)))

                return predicted_class, R
            return None
        predicted_class = counts.argmax()

                # Calculate certified radius
        pA = nA / (nA+nB)
        R = self.sigma * (np.sqrt(2) * (np.sqrt(nA + nB) * np.abs(pA - 0.5)))
        return predicted_class , R

    def certify(self, x, n0, alpha=0.05):
        """Certify the robustness of the model around x."""
        counts0 = self.sample_under_noise(x)
        c_hat_A = counts0.argmax()

        counts = self.sample_under_noise(x)
        pA = self.lower_conf_bound(counts[c_hat_A], counts.sum(), 1 - alpha)

        if pA > 0.5:  # Change 0.5 to the value you want
            return c_hat_A, self.sigma * np.sqrt(-2 * np.log(1 - pA))
        else:
            return None
