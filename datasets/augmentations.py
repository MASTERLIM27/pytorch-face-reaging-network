import albumentations as A
from albumentations.pytorch import ToTensorV2

transform_normalize = A.ReplayCompose([
    A.Resize(512, 512),
    # Crop is really important
    A.RandomCrop(256, 256),
    A.ColorJitter(p=0.8),
    A.Blur(p=0.1),
    A.RandomBrightnessContrast(),
    A.Affine(rotate=[-30, 30],scale=(0.5,1.5), p=0.8),
    ToTensorV2(),
])

transform_no_crop = A.ReplayCompose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.3),
    A.Rotate(limit=5, p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.02,
        scale_limit=0.05,
        rotate_limit=5,
        p=0.5
    ),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    ToTensorV2(),
])
