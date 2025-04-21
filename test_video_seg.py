import os
import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp
from models.discriminator import PatchGANDiscriminator
from models.generator import Generator
from training.trainer import FRAN

# Device config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants
COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 204, 204],
    2: [204, 255, 204],
    3: [204, 204, 255],
    4: [255, 255, 153],
    5: [255, 153, 255],
    6: [153, 255, 255],
}

# MediaPipe setup
def create_segmenter():
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path="models/selfie_multiclass_256x256.tflite"),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True,
    )
    return ImageSegmenter.create_from_options(options)

segmenter = create_segmenter()

# Load model
def load_model_from_checkpoint(checkpoint_path, generator):
    model = FRAN(generator, PatchGANDiscriminator())
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.generator.to(device)

generator_model = load_model_from_checkpoint(
    checkpoint_path="./logs/model/version_23/model_checkpoints/last.ckpt",
    generator=Generator()
)

# Image transform
IMAGE_TRANSFORM = A.Compose([
    A.Resize(512, 512),
    ToTensorV2(),
])

# Generate age map using segmentation
def generate_age_map(image, input_age, output_age):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask.numpy_view()

    category_mask_resized = cv2.resize(category_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    age_map = np.full((512, 512), input_age / 100, dtype=np.float32)
    age_map[(category_mask_resized == 1) | (category_mask_resized == 2) | (category_mask_resized == 3)] = output_age / 100

    age_map_tensor = torch.tensor(age_map).unsqueeze(0).to(device)
    return age_map_tensor

# Apply model

def generate_segment_mask(image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask.numpy_view()
    category_mask_resized = cv2.resize(category_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    return category_mask_resized

import os

os.makedirs("debug", exist_ok=True)  # Create debug folder if it doesn't exist
frame_count = 0  # Global frame index

def age_filter(image, input_age, output_age):
    global frame_count

    original_np = np.array(image.convert("RGB"))
    segmentation_mask = generate_segment_mask(image)

    # Categories to age (face, hair, body)
    mask = (segmentation_mask == 1) | (segmentation_mask == 2) | (segmentation_mask == 3)

    # Resize for model
    resized_image = cv2.resize(original_np, (512, 512))
    resized_mask = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)

    # Mask the input image
    masked_image = resized_image * resized_mask[:, :, np.newaxis]

    # Transform for model
    transformed = IMAGE_TRANSFORM(image=masked_image)
    normalized_input = transformed["image"].to(device) / 127.5 - 1

    source_age = torch.full((1, 512, 512), input_age / 100.0, device=device)
    target_age = torch.full((1, 512, 512), input_age / 100.0, device=device)
    target_age[0][(resized_mask == 1)] = output_age / 100.0

    model_input = torch.cat([normalized_input, source_age, target_age], dim=0).unsqueeze(0)

    with torch.no_grad():
        residual = generator_model(model_input).squeeze(0)

    predicted = (normalized_input + residual).clamp(-1, 1)
    predicted_np = predicted.cpu().permute(1, 2, 0).numpy()
    predicted_img = ((predicted_np * 0.5 + 0.5) * 255).astype(np.uint8)

    # Composite output with original
    composite = resized_image.copy()
    composite[resized_mask == 1] = predicted_img[resized_mask == 1]

    # Debug: Save input/output images
    input_debug = ((normalized_input.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    cv2.imwrite(f"debug/input_{frame_count:04d}.png", cv2.cvtColor(input_debug, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"debug/output_{frame_count:04d}.png", cv2.cvtColor(predicted_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"debug/composite_{frame_count:04d}.png", cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    # Optional: display instead of save
    # cv2.imshow("Input", input_debug)
    # cv2.imshow("Output", predicted_img)
    # cv2.imshow("Composite", composite)
    # cv2.waitKey(1)

    frame_count += 1
    return Image.fromarray(composite)


# Process video
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

# Example usage
process_video("./example-images/video_4.mp4", "./example-images/output_seg_4_23_age_50_80.mp4", 50, 80)