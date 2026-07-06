import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.xception_net import load_xception_model
from models.meso_net import load_meso_model

def get_data_loaders(data_dir, batch_size, input_size):
    """
    Load data from a standard ImageFolder directory structure:
    data_dir/
      train/
        real/
        fake/
      val/
        real/
        fake/
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise ValueError(f"Dataset directory must contain 'train' and 'val' subdirectories with 'real' and 'fake' classes.")

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # Handle different loss functions based on model output shape
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                labels_float = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels_float)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                corrects = torch.sum(preds == labels_float)
            else:
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == labels.data)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += corrects.item()
            total_samples += inputs.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    labels_float = labels.float().unsqueeze(1)
                    loss = criterion(outputs, labels_float)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    corrects = torch.sum(preds == labels_float)
                else:
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    corrects = torch.sum(preds == labels.data)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += corrects.item()
                val_total_samples += inputs.size(0)
                
        val_loss = val_running_loss / val_total_samples
        val_acc = val_running_corrects / val_total_samples
        
        print(f"Val Loss:   {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    print("\nTraining complete!")
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Deepfake Detection Models")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory (must contain 'train' and 'val' subdirectories)")
    parser.add_argument("--model", type=str, required=True, choices=["xception", "meso"], help="Model architecture to fine-tune")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save trained model weights")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model, loss, and input size
    if args.model == "xception":
        print("Initializing CustomXception model...")
        model = load_xception_model(use_custom=True)
        criterion = nn.CrossEntropyLoss()
        input_size = 224
    else:
        print("Initializing Meso4 model...")
        model = load_meso_model(model_type='meso4')
        criterion = nn.BCEWithLogitsLoss()
        input_size = 256
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(f"Loading dataset from {args.data_dir}...")
    try:
        train_loader, val_loader, classes = get_data_loaders(args.data_dir, args.batch_size, input_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Classes found: {classes}")
    
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        model_name=args.model,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()
