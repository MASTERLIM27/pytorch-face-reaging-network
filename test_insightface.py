import torch
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import torch.nn.functional as F
import albumentations as A
from datasets.augmentations import transform_no_crop

# ----------- Init InsightFace (ArcFace) ----------- #
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

# ----------- Helper: Get face embedding from in-memory image ----------- #
def get_embedding_from_image(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (112, 112))
    faces = app.get(image_rgb)
    if len(faces) == 0:
        print("No face found.")
        return None, image_rgb
    emb = torch.tensor(faces[0].embedding, dtype=torch.float32)
    return emb, image_rgb

# ----------- Load and augment image ----------- #
# ./logs/model/version_24/output_images/step_5500/augmented_1_input.png
img_path = "./logs/model/version_37/output_images/step_1000/augmented_1_output.png"
image = cv2.imread(img_path)  # BGR format, dtype=uint8
augmented_result = transform_no_crop(image=image)
image_tensor = augmented_result["image"]
image = image_tensor.permute(1, 2, 0).cpu().numpy()

# Augment the image using albumentations
augmented_result = transform_no_crop(image=image)
aug_tensor = augmented_result["image"]
aug_image = aug_tensor.permute(1, 2, 0).cpu().numpy()
# aug_image_path = "./example-images/output_example_img.png"
# aug_image = cv2.imread(aug_image_path)

# ----------- Get embeddings ----------- #
emb1, img1_vis = get_embedding_from_image(image)
emb2, img2_vis = get_embedding_from_image(aug_image)

# ----------- Identity Loss ----------- #
if emb1 is not None and emb2 is not None:
    emb1 = F.normalize(emb1.unsqueeze(0), dim=1)
    emb2 = F.normalize(emb2.unsqueeze(0), dim=1)
    identity_loss = 1 - F.cosine_similarity(emb1, emb2).item()
    print(f"Identity Loss (1 - cosine similarity): {identity_loss:.4f}")
else:
    identity_loss = None
    print("Embedding extraction failed.")

# ----------- Visualization ----------- #
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img1_vis)
axs[0].set_title("Original")
axs[0].axis('off')
axs[1].imshow(img2_vis)
axs[1].set_title("Augmented")
axs[1].axis('off')
plt.suptitle(f"Identity Loss: {identity_loss:.4f}" if identity_loss is not None else "Face not found")
plt.show()
