import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_loader import OriginalXBDDataset
from model import UNetXBD

def train_xbd():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 1. Initialize Dataset & Dataloader
    # Pointing to the raw pre-post images directory you provided
    IMAGE_DIR = r'F:\NSU\cse499 a-b\xview\raw pre-post images'
    MASK_DIR = r'F:\NSU\cse499 a-b\xview\masks_png'
    
  
    
    # 2. Initialize Model
    model = UNetXBD(num_classes=3).to(device)
    
    # 3. Loss & Optimizer (As per paper)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=5e-4,           # Learning rate from paper
        weight_decay=1e-4  # Weight decay from paper
    )
    
    # 4. Scheduler (Cosine Annealing over 40 epochs)
    num_epochs = 40
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Simulated Dataloader Loop
        # Replace with: for images, masks in dataloader:
        images = torch.rand((4, 3, 1024, 1024)).to(device) 
        masks = torch.randint(0, 3, (4, 1024, 1024)).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Step learning rate scheduler
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

if __name__ == "__main__":
    train_xbd()
