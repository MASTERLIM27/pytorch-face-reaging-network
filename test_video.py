import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.discriminator import PatchGANDiscriminator
from models.generator import Generator
from training.trainer import FRAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_from_checkpoint(checkpoint_path, generator):
    model = FRAN(generator, PatchGANDiscriminator())
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.generator

generator_model = load_model_from_checkpoint(
    checkpoint_path="./logs/model/version_24/model_checkpoints/last.ckpt",
    generator=Generator()
).to(device)

# Load model
# generator_model = Generator()
# generator_model.load_state_dict(torch.load('./models/pretrained_model/large-aging-model.h5',map_location=torch.device(device)))
# generator_model.to(device)
# generator_model.eval()

def age_filter(image, input_age, output_age):
    transform_resize = A.Compose([
        A.Resize(512, 512),
        ToTensorV2(),
    ])
    
    image_np = np.array(image.convert("RGB"))
    transformed = transform_resize(image=image_np)
    normalized_input = transformed["image"].to(device) / 127.5 - 1
    
    source_age = torch.full((1, 512, 512), input_age / 100.0, device=device)
    target_age = torch.full((1, 512, 512), output_age / 100.0, device=device)
    
    model_input = torch.cat([normalized_input, source_age, target_age], dim=0).unsqueeze(0)
    
    with torch.no_grad():
        residual = generator_model(model_input).squeeze(0)
    
    predicted = (normalized_input + residual).clamp(-1, 1)
    predicted = predicted.cpu().permute(1, 2, 0).numpy()
    predicted = ((predicted * 0.5 + 0.5) * 255).astype(np.uint8)
    
    return Image.fromarray(predicted)

def process_video(input_video_path, output_video_path, input_age, output_age):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        aged_image = age_filter(image, input_age, output_age)
        
        aged_frame = cv2.cvtColor(np.array(aged_image), cv2.COLOR_RGB2BGR)
        aged_frame = cv2.resize(aged_frame, (frame_width, frame_height))
        
        out.write(aged_frame)
    
    cap.release()
    out.release()
    print("Processing complete. Output saved at", output_video_path)


process_video("./example-images/video_4.mp4", "./example-images/output_24_age_50_20.mp4", 50, 20)
