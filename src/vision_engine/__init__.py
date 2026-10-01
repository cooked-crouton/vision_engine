import timm
import torch
from PIL import Image
from torchvision import transforms

# 1. Créer l'encodeur sans la couche de classification finale (num_classes=0)
model = timm.create_model('resnet50', pretrained=True, num_classes=0)
model.eval()

# 2. Obtenir les transformations spécifiques au modèle
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# 3. Charger et encoder l'image
img = Image.open('tests/votre_image.jpg')
input_tensor = transform(img).unsqueeze(0) # Ajouter la dimension de batch

with torch.no_grad():
    image_embedding = model(input_tensor)

print(image_embedding.shape) # Résultat: [1, 2048] (vecteur de caractéristiques)

def main() -> None:
    print("Hello from vision-engine!")
