import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create directory for saving results
os.makedirs('results', exist_ok=True)

# Hyperparameters
batch_size = 128
learning_rates = [1e-3, 1e-2, 1e-1]  # Learning rates to test
num_epochs = 10
input_size = 784  # 28x28
hidden_size1 = 128
hidden_size2 = 64
output_size = 10

# Load MNIST dataset
def load_mnist():
    from torchvision import datasets, transforms
    import torch
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )
    
    # Convert to numpy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0  # Normalize to [0,1]
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_test = test_dataset.targets.numpy()
    
    return (X_train, y_train), (X_test, y_test)

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Loss function (Cross-Entropy)
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-15)
    return np.sum(log_likelihood) / m

# Neural Network with NumPy
class FeedForwardNet:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize a 3-layer neural network (784-128-10) as per the architecture diagram.
        """
        # Initialize weights using He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Output layer weights (hidden_size to output_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = softmax(self.z2)  # Softmax for multi-class classification
        
        return self.output
    
    def backward(self, X, y, output, learning_rate, optimizer):
        m = X.shape[0]  # Number of samples in the batch
        
        # Calculate gradient of loss with respect to z2 (output layer)
        # For cross-entropy loss with softmax: dL/dz2 = y_pred - y_true
        dz2 = output.copy()
        dz2[range(m), y] -= 1  # y is the true class index
        dz2 /= m  # Average over batch
        
        # Gradients for output layer (W2 and b2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Backpropagate to hidden layer
        da1 = np.dot(dz2, self.W2.T)  # Gradient w.r.t. a1
        dz1 = da1 * relu_derivative(self.z1)  # Gradient w.r.t. z1
        
        # Gradients for hidden layer (W1 and b1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update parameters using the optimizer
        optimizer.update_params(
            [self.W1, self.W2],  # Weights
            [self.b1, self.b2],  # Biases
            [dW1, dW2],         # Weight gradients
            [db1, db2],         # Bias gradients
            learning_rate
        )

# Standard SGD Optimizer (without momentum)
class SGD:
    def __init__(self):
        pass  # No parameters needed for standard SGD
        
    def update_params(self, weights, biases, dW, db, learning_rate):
        # Update parameters using standard SGD: θ_{t+1} = θ_t - η * ∇L(θ_t)
        for i in range(len(weights)):
            weights[i] -= learning_rate * dW[i]
            biases[i] -= learning_rate * db[i]

# Adam Optimizer
class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1      # β₁ in the formula (for first moment)
        self.beta2 = beta2      # β₂ in the formula (for second moment)
        self.epsilon = epsilon  # ε in the formula (for numerical stability)
        self.m_w = []           # First moment vector for weights
        self.v_w = []           # Second moment vector for weights
        self.m_b = []           # First moment vector for biases
        self.v_b = []           # Second moment vector for biases
        self.t = 0              # Time step
        
    def update_params(self, weights, biases, dW, db, learning_rate):
        # Initialize moment vectors on first call
        if not self.m_w:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        self.t += 1
        
        for i in range(len(weights)):
            # Update biased first moment estimate for weights: m_t = β₁ * m_{t-1} + (1-β₁) * ∇L(θ_t)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            # Update biased second raw moment estimate for weights: v_t = β₂ * v_{t-1} + (1-β₂) * (∇L(θ_t))^2
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dW[i] ** 2)
            
            # Compute bias-corrected first moment estimate: m̂_t = m_t / (1 - β₁^t)
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate: v̂_t = v_t / (1 - β₂^t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            
            # Update weights: θ_{t+1} = θ_t - (η / (√(v̂_t) + ε)) * m̂_t
            weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            
            # Update biased first moment estimate for biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
            # Update biased second raw moment estimate for biases
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db[i] ** 2)
            
            # Compute bias-corrected first moment estimate for biases
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate for biases
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            # Update biases
            biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

def train_model(model, X_train, y_train, X_val, y_val, optimizer, learning_rate, num_epochs, batch_size):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    n_batches = int(np.ceil(len(X_train) / batch_size))
    
    for epoch in range(num_epochs):
        # Shuffle the data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for i in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            output = model.forward(X_batch)
            
            # Compute loss
            loss = cross_entropy_loss(output, y_batch)
            epoch_loss += loss
            
            # Backward pass and update weights
            model.backward(X_batch, y_batch, output, learning_rate, optimizer)
            
            # Calculate accuracy
            predictions = np.argmax(output, axis=1)
            correct += np.sum(predictions == y_batch)
            total += len(y_batch)
        
        # Calculate training metrics
        avg_train_loss = epoch_loss / n_batches
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)
        
        # Validation
        val_output = model.forward(X_val)
        val_loss = cross_entropy_loss(val_output, y_val)
        val_predictions = np.argmax(val_output, axis=1)
        val_accuracy = 100 * np.mean(val_predictions == y_val)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, ' \
              f'Train Loss: {avg_train_loss:.4f}, ' \
              f'Train Acc: {train_accuracy:.2f}%, ' \
              f'Val Loss: {val_loss:.4f}, ' \
              f'Val Acc: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

def plot_results(results, filename='results/training_results.png'):
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    for label, (train_loss, val_loss, _, _) in results.items():
        plt.plot(train_loss, label=f'{label} Train')
        plt.plot(val_loss, '--', label=f'{label} Val')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    for label, (_, _, train_acc, val_acc) in results.items():
        plt.plot(train_acc, label=f'{label} Train')
        plt.plot(val_acc, '--', label=f'{label} Val')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist()
    
    # Split into training and validation sets (90% train, 10% validation)
    split_idx = int(0.9 * len(X_train))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    # Dictionary to store results
    results = {}
    
    # Test different optimizers and learning rates
    for lr in learning_rates:
        print(f"\n{'='*50}")
        print(f"Training with learning rate: {lr}")
        print(f"{'='*50}")
        
        # Test SGD
        print("\nTraining with SGD...")
        model_sgd = FeedForwardNet(input_size, hidden_size1, hidden_size2, output_size)
        sgd_optimizer = SGD()  # No momentum parameter for standard SGD
        train_loss_sgd, val_loss_sgd, train_acc_sgd, val_acc_sgd = train_model(
            model_sgd, X_train, y_train, X_val, y_val, sgd_optimizer, lr, num_epochs, batch_size
        )
        results[f'SGD (lr={lr})'] = (train_loss_sgd, val_loss_sgd, train_acc_sgd, val_acc_sgd)
        
        # Test Adam
        print("\nTraining with Adam...")
        model_adam = FeedForwardNet(input_size, hidden_size1, hidden_size2, output_size)
        adam_optimizer = Adam()
        train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam = train_model(
            model_adam, X_train, y_train, X_val, y_val, adam_optimizer, lr, num_epochs, batch_size
        )
        results[f'Adam (lr={lr})'] = (train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam)
    
    # Plot results
    plot_results(results)
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    # Test the best model (you can modify this to test all models)
    test_output = model_adam.forward(X_test)
    test_loss = cross_entropy_loss(test_output, y_test)
    test_predictions = np.argmax(test_output, axis=1)
    test_accuracy = 100 * np.mean(test_predictions == y_test)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
