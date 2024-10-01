import torch
import clip
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu" # Sur quoi on store les données, si les données sont pas storées au même endroit 
# que les tenseurs ca pose pb, donc en gros on rajoute un .to(device) a chaque fois pour tout mettre au mm endroit et on fait en sorte
# de prioriser le gpu so y en a un (cuda)
model, preprocess = clip.load("ViT-B/32", device=device) # on importe le modele et la fonction qui nous permet de preprocess les images 
# = de les transformer en vecteur interpretable par le modele


image = preprocess(Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)).unsqueeze(0).to(device) # Je vais chercher l'image d'internet psque j'en avais pas sous la main, mais c'est un chat en gros 

text = clip.tokenize(["a diagram", "a dog", "a cat", "a panda"]).to(device) # Les labels potentiels

with torch.no_grad():
    image_features = model.encode_image(image) # encode l'image en embedding
    text_features = model.encode_text(text) # idem pour les labels
    
    logits_per_image, logits_per_text = model(image, text) # calcule les logits (log counts)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy() # softmax pour avoir des probas

print("Label probs:", probs)
# Ca printe Label probs: [[0.156    0.00937  0.8306   0.004093]] donc en gros le modèle prédit 15.6% d'etre un diagrame, 83% d'etre un chat etc

