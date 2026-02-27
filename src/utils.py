import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import os
from typing import Tuple, List
import numpy as np


def creating_datasets(root_dir='../data/raw/', test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train, validation, and test datasets
    Args:
        root_dir: Path to the root directory containing the image folders
        test_size: Proportion of the test split
        val_size: Proportion of the validation split
        random_state: Random seed 
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_transform = T.Compose([
        T.Resize((128, 128)),
        T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.2),
        T.RandomRotation(degrees=(15, 30)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = datasets.ImageFolder(root=root_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=root_dir, transform=test_transform)
    val_dataset = datasets.ImageFolder(root=root_dir, transform=test_transform)
    
    targets = train_dataset.targets
    indices = list(range(len(targets)))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=targets, random_state=random_state
    )
    val_size_adjusted = val_size / (1 - test_size)
    train_val_targets = [targets[i] for i in train_idx]
    
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size_adjusted, stratify=train_val_targets, random_state=random_state
    )
    
    train_dataset = Subset(train_dataset, train_idx)
    test_dataset = Subset(test_dataset, test_idx)
    val_dataset = Subset(val_dataset, val_idx)
    
    return train_dataset, val_dataset, test_dataset


def load_custom_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess a custom image for prediction
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image tensor ready for model input (1, 3, 128, 128)
    """
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0) 
    
    return image_tensor


def predict_custom_images(model, images: List[torch.Tensor], device: str = 'cpu') -> Tuple[List[int], np.ndarray]:
    """
    Predict labels for custom images
    
    Args:
        model: Trained model
        images: List of preprocessed image tensors
        device: Device to run
    
    Returns:
        Tuple of (predicted_class_indices, probabilities_array)
    """
    model.eval()
    predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for img_tensor in images:
            img_tensor = img_tensor.to(device)
            output = model(img_tensor)
            
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted = torch.argmax(probs, dim=1)
            
            predictions.append(predicted.item())
            all_probabilities.append(probs.cpu().numpy()[0])
    
    return predictions, np.array(all_probabilities)


def load_custom_test_folder(folder_path: str, class_names: List[str]) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    Load custom test images from a folder structure:
    folder_path/
        paper/
        rock/
        scissors/
    
    Args:
        folder_path: Path to folder
        class_names: List of class names (e.g., ['Paper', 'Rock', 'Scissors'])
    
    Returns:
        Tuple of (images, true_labels, image_paths)
    """
    images = []
    image_paths = []
    true_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name.lower())
        if os.path.exists(class_folder):
            for img_file in os.listdir(class_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_folder, img_file)
                    image_paths.append(img_path)
                    images.append(load_custom_image(img_path))
                    true_labels.append(class_idx)
    
    return images, true_labels, image_paths