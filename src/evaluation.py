import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            precision_recall_fscore_support, roc_curve, 
                            auc, roc_auc_score)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Tuple


def get_predictions(model: nn.Module,
                   data_loader: DataLoader,
                   device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get model predictions and true labels for a dataset
    
    Returns:
        Tuple of (predictions, true_labels)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_predictions), np.array(all_labels)


def get_predictions_with_probabilities(model: nn.Module,
                                      data_loader: DataLoader,
                                      device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions, probabilities and true labels for a dataset
    
    Returns:
        Tuple of (predictions, probabilities, true_labels)
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: List[str],
                         save_path: Optional[str] = None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_classification_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 class_names: List[str]):
    """Print detailed classification metrics"""
    
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   digits=4)
    print(report)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    from sklearn.metrics import accuracy_score
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_roc_curves(y_true: np.ndarray,
                   y_probabilities: np.ndarray,
                   class_names: List[str],
                   save_path: Optional[str] = None):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: True labels
        y_probabilities: Predicted probabilities for each class
        class_names: Names of classes
        save_path: Path to save the figure
    """
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', lw=3,
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return roc_auc


def plot_misclassified_samples(images: np.ndarray,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_probabilities: np.ndarray,
                              class_names: List[str],
                              max_samples: int = 12,
                              save_path: Optional[str] = None):
    """
    Plot misclassified samples with their predictions and confidence
    
    Args:
        images: Array of images (N, H, W, C) or (N, C, H, W)
        y_true: True labels
        y_pred: Predicted labels
        y_probabilities: Prediction probabilities
        class_names: Names of classes
        max_samples: Maximum number of samples to display
        save_path: Path to save the figure
    """
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified samples found!")
        return
    
    num_samples = min(len(misclassified_idx), max_samples)
    selected_idx = misclassified_idx[:num_samples]
    
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_idx):
        img = images[idx]
        if img.shape[0] == 3 or img.shape[0] == 1:  
            img = np.transpose(img, (1, 2, 0))
        
        if img.shape[2] == 1:
            img = img.squeeze()
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        confidence = y_probabilities[idx, y_pred[idx]]
        
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}"
        axes[i].set_title(title, fontsize=10, color='red', fontweight='bold')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Misclassified Samples (Total: {len(misclassified_idx)})', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_misclassifications(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_probabilities: np.ndarray,
                               class_names: List[str]) -> Dict:
    """
    Perform detailed analysis of misclassifications
    
    Returns:
        Dictionary containing misclassification statistics
    """
    misclassified_idx = np.where(y_true != y_pred)[0]
    n_total = len(y_true)
    n_misclassified = len(misclassified_idx)
    
    print(f"Total samples: {n_total}")
    print(f"Misclassified: {n_misclassified} ({n_misclassified/n_total*100:.2f}%)")
    print(f"Correctly classified: {n_total - n_misclassified} ({(n_total - n_misclassified)/n_total*100:.2f}%)")
    
    
    
    confusion_counts = {}
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                count = np.sum((y_true == i) & (y_pred == j))
                if count > 0:
                    pair = f"{class_names[i]} → {class_names[j]}"
                    confusion_counts[pair] = count
                    print(f"  {pair}: {count} samples")
    
    print()
    
    correct_idx = np.where(y_true == y_pred)[0]
    correct_confidences = [y_probabilities[i, y_pred[i]] for i in correct_idx]
    misclass_confidences = [y_probabilities[i, y_pred[i]] for i in misclassified_idx]
    
    print(f"Per-Class Misclassification Rate:")
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_total = np.sum(class_mask)
        class_misclassified = np.sum((y_true == i) & (y_pred != i))
        
        if class_total > 0:
            misclass_rate = class_misclassified / class_total * 100
            print(f"  {class_name}: {class_misclassified}/{class_total} ({misclass_rate:.2f}%)")
    
    
    return {
        'total_samples': n_total,
        'misclassified': n_misclassified,
        'confusion_pairs': confusion_counts,
        'correct_confidence': correct_confidences,
        'misclass_confidence': misclass_confidences
    }