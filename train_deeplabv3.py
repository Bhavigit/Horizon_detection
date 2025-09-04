import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 1. CONFIG
DATA_DIR = "/content/horizon_dataset_1"
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
BATCH_SIZE = 8
NUM_CLASSES = 2
EPOCHS = 300
PATIENCE = 25
ENCODER_NAME = "resnet101"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 2. DATASET CLASS
class SegDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        print(f"DEBUG: Checking for image directory: {self.img_dir}")
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        print(f"DEBUG: Checking for mask directory: {self.mask_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        self.images = sorted(os.listdir(img_dir))
        print(f"Initialized dataset with {len(self.images)} images from {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask = mask.squeeze(-1)
            elif mask.shape[2] == 3:
                mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()

        if idx == 0:
            mask_np_values = np.unique(mask.cpu().numpy())
            print(f"DEBUG (SegDataset): Mask unique values for first image ({self.images[idx]}): {mask_np_values}")
            if 1 in mask_np_values:
                print("DEBUG (SegDataset): Confirmed: Class 1 (horizon) IS present in a training mask.")
            else:
                print("DEBUG (SegDataset): WARNING: Class 1 (horizon) NOT found in the first training mask. Check mask preprocessing!")

        return image, mask


# 3. HELPER FUNCTIONS
def get_transforms():
    "Returns data augmentation and normalization transforms."
    train_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomGamma(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transforms, val_transforms


def get_dataloaders(train_transforms, val_transforms):
    "Initializes datasets, calculates class weights, and returns data loaders."
    train_dataset = SegDataset(
        img_dir=os.path.join(IMAGE_DIR, 'train'),
        mask_dir=os.path.join(MASK_DIR, 'train'),
        transform=train_transforms
    )
    val_dataset = SegDataset(
        img_dir=os.path.join(IMAGE_DIR, 'val'),
        mask_dir=os.path.join(MASK_DIR, 'val'),
        transform=val_transforms
    )
    print(f"\nTotal training images loaded: {len(train_dataset)}")
    print(f"Total validation images loaded: {len(val_dataset)}\n")
    print("Calculating class frequencies for loss weighting...")

    class_pixel_counts = Counter()
    raw_train_dataset = SegDataset(
        img_dir=os.path.join(IMAGE_DIR, 'train'),
        mask_dir=os.path.join(MASK_DIR, 'train'),
        transform=None
    )
    for i in tqdm(range(len(raw_train_dataset)), desc="Counting pixels"):
        _, mask = raw_train_dataset[i]
        class_pixel_counts.update(mask.flatten().tolist())
    total_pixels = sum(class_pixel_counts.values())

    if total_pixels == 0:
        class_weights = torch.ones(NUM_CLASSES).to(DEVICE)
        print("WARNING: No pixels found in masks, using uniform weights.")
    else:
        weights = [0] * NUM_CLASSES
        for cls in range(NUM_CLASSES):
            count = class_pixel_counts.get(cls, 0)
            if count == 0:
                weights[cls] = 1.0
                print(f"WARNING: Class {cls} has 0 pixels. Assigning weight 1.0.")
            else:
                weights[cls] = total_pixels / count
        class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        print(f"Class pixel counts: {class_pixel_counts}")
        print(f"Calculated class weights: {class_weights}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    return train_loader, val_loader, class_weights


def create_model(class_weights, encoder_name):
    "Initializes the model, loss functions, optimizer, and scheduler."
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)

    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    return model, ce_loss, dice_loss, optimizer, scheduler


def train_one_epoch(model, loader, ce_loss, dice_loss, optimizer):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        preds = model(images)
        loss = (ce_loss(preds, masks) * 0.5) + (dice_loss(preds[:, 1, :, :], (masks == 1).float()) * 0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_one_epoch(model, loader, ce_loss, dice_loss):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images)
            loss = (ce_loss(preds, masks) * 0.5) + (dice_loss(preds[:, 1, :, :], (masks == 1).float()) * 0.5)
            val_loss += loss.item()
    return val_loss / len(loader)


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    print("Loss curve saved as 'loss_curve.png'")
    plt.close()


def main():
    train_transforms, val_transforms = get_transforms()
    train_loader, val_loader, class_weights = get_dataloaders(train_transforms, val_transforms)
    model, ce_loss, dice_loss, optimizer, scheduler = create_model(class_weights, ENCODER_NAME)

    writer = SummaryWriter('runs/deeplabv3_experiment_1')
    print(f"Starting training on {DEVICE} for {EPOCHS} epochs...\n")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, ce_loss, dice_loss, optimizer)
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(f" Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")

        avg_val_loss = validate_one_epoch(model, val_loader, ce_loss, dice_loss)
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        print(f" Epoch {epoch + 1} Val Loss:   {avg_val_loss:.4f}\n")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"deeplabv3plus_horizon_best.pth")
            print(f" Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f" Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs. Training stopped.")
            break

        if (epoch + 1) % 10 == 0:
            epoch_model_path = f"deeplabv3plus_horizon_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), epoch_model_path)
            print(f" Saved model for epoch {epoch+1} as {epoch_model_path}")

    writer.close()
    torch.save(model.state_dict(), f"deeplabv3plus_horizon_last.pth")
    print("Model saved (last epoch) as deeplabv3plus_horizon_last.pth")
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
