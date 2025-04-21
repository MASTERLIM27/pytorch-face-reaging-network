import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

from models.discriminator import PatchGANDiscriminator
from models.generator import Generator
from training.trainer import FRAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_from_checkpoint(checkpoint_path, generator):
    model = FRAN(generator, PatchGANDiscriminator())
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.generator

# Preprocessing pipeline
transform_resize = A.Compose([
    A.Resize(512, 512),
    ToTensorV2(),
])

# Load model
# generator_model = Generator()
# generator_model.load_state_dict(torch.load('./models/pretrained_model/large-aging-model.h5', map_location=torch.device(device)))
# generator_model.to(device)
# generator_model.eval()

# fran-step=810000.ckpt
# Load the trained generator model
generator_model = load_model_from_checkpoint(
    checkpoint_path="./logs/model/version_24/model_checkpoints/last.ckpt",
    generator=Generator(),
).to(device)
# generator_model.eval()

def age_filter(image, input_age, output_age):
    # Convert and preprocess image
    image_np = np.array(image.convert("RGB"))
    transformed = transform_resize(image=image_np)
    normalized_input = transformed["image"].to(device) / 127.5 - 1
    
    # Create age maps
    source_age = torch.full((1, 512, 512), input_age/100.0, device=device)
    target_age = torch.full((1, 512, 512), output_age/100.0, device=device)
    
    # Create full input tensor
    model_input = torch.cat([
        normalized_input,
        source_age,
        target_age
    ], dim=0).unsqueeze(0)

    # Generate output
    with torch.no_grad():
        residual = generator_model(model_input).squeeze(0).clamp(-1, 1)

    print(f"Input Min-Max: {normalized_input.min().item()}, {normalized_input.max().item()}")
    print(f"Residual Min-Max: {residual.min().item()}, {residual.max().item()}")
    
    # Combine with normalized input and denormalize
    predicted = (normalized_input + residual).clamp(-1, 1)
    predicted = predicted.cpu().permute(1, 2, 0).numpy()
    predicted = ((predicted + 1) * 127.5).clip(0, 255).astype(np.uint8)

    return Image.fromarray(predicted).resize(image.size)

def process_and_show_image(image_path, input_age, output_age):
    image = Image.open(image_path).convert("RGB")
    output_image = age_filter(image, input_age, output_age)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"Input (Age: {input_age})")
    axes[0].axis("off")

    axes[1].imshow(output_image)
    axes[1].set_title(f"Output (Age: {output_age})")
    axes[1].axis("off")
    plt.show()

# example-images/input_example_img.png
# example-images/18_seed0002.png
process_and_show_image("/home/andre/fran/example-images/soe.png", 20, 83)