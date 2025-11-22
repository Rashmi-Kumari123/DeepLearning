#!/usr/bin/env python3
"""
MNIST Optimization Study: Comparing SGD and Adam Optimizers

This script implements a comprehensive comparison between SGD and Adam optimizers
on the MNIST dataset using different neural network architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)

# Check for CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for saving results
os.makedirs('results', exist_ok=True)

# Define hyperparameters
batch_size = 128
learning_rates = [1e-3, 1e-2, 1e-1]  # Learning rates to test
num_epochs = 10

# Define data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Function to load MNIST dataset
def load_mnist():
    """Load and return MNIST dataset with train/test loaders."""
    print("Loading MNIST dataset...")
    
    # Download and load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader

# Define Feedforward Neural Network
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Define LeNet CNN
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 7*7*64)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to reset model weights
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs, model_name):
    """Train the model and return training metrics."""
    model.train()
    train_losses = []
    train_acc = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        # Wrap train_loader with tqdm for progress bar
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_iter.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {epoch_loss:.4f}, '
              f'Accuracy: {epoch_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')
    
    return train_losses, train_acc

# Evaluation function
def evaluate_model(model, test_loader, criterion):
    """Evaluate the model on test data and return metrics."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    return test_loss, test_acc

# Function to plot and save results
def plot_results(results, filename='results/training_results.png'):
    """Plot training and test results."""
    plt.figure(figsize=(15, 10))
    
    # Plot training accuracy
    plt.subplot(2, 1, 1)
    for key, data in results.items():
        if 'Adam' in key:  # Only plot Adam for clarity
            plt.plot(data['train_acc'], '--', label=f'{key} (Train)')
            plt.plot([data['test_acc']] * len(data['train_acc']), '-', 
                    label=f'{key} (Test)')
    
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot training loss
    plt.subplot(2, 1, 2)
    for key, data in results.items():
        if 'Adam' in key:  # Only plot Adam for clarity
            plt.plot(data['train_losses'], label=key)
    
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Results saved to {filename}")

# Main function
def main():
    """Main function to run the MNIST optimization study."""
    # Load data
    train_loader, test_loader = load_mnist()
    
    # Initialize models
    models = {
        'FFN': FeedForwardNet().to(device),
        'LeNet': LeNet().to(device)
    }
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    # Run experiments
    for model_name, model in models.items():
        print(f'\n=== Training {model_name} ===')
        
        for lr in learning_rates:
            print(f'\nLearning Rate: {lr}')
            
            # Try both SGD and Adam optimizers
            optimizers = {
                'SGD': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
                'Adam': optim.Adam(model.parameters(), lr=lr)
            }
            
            for opt_name, optimizer in optimizers.items():
                print(f'\nOptimizer: {opt_name}')
                
                # Reset model weights
                model.apply(weight_reset)
                
                # Train and evaluate
                train_losses, train_acc = train_model(
                    model, train_loader, criterion, optimizer, 
                    num_epochs, f'{model_name}_{opt_name}'
                )
                
                test_loss, test_acc = evaluate_model(model, test_loader, criterion)
                
                # Store results
                key = f'{model_name}_{opt_name}_lr{lr}'
                results[key] = {
                    'train_losses': train_losses,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc
                }
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"{'Model':<10} {'Optimizer':<8} {'LR':<10} {'Test Acc':<10}")
    print("-" * 40)
    
    for key, data in results.items():
        parts = key.split('_')
        model_name = parts[0]
        opt_name = parts[1]
        lr = parts[2][2:]  # Remove 'lr' prefix
        
        print(f"{model_name:<10} {opt_name:<8} {lr:<10} {data['test_acc']:.2f}%")
    
    # Plot and save results
    plot_results(results)

if __name__ == '__main__':
    main()
