"""
SVM Classifier implementation from scratch using NumPy.
Implements binary linear SVM with gradient descent optimization.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import (SVM_DIR, get_timestamped_filename)

class LinearSVM:
    """
    Linear Support Vector Machine classifier implemented from scratch.
    """
    def __init__(self, learning_rate=0.01, regularization=1.0, epochs=1000):
        """
        Initialize SVM classifier.
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            regularization (float): Regularization parameter (C)
            epochs (int): Number of training epochs
        """
        self.learning_rate = learning_rate
        self.regularization = regularization  # C parameter
        self.epochs = epochs
        self.w = None  # weight vector
        self.b = 0     # bias term
        self.feature_means = None
        self.feature_stds = None
        self.loss_history = []
        self.support_vectors = None
        
    def normalize_features(self, X):
        """
        Normalize features using means and standard deviations from training.
        Manual implementation without using NumPy's statistical functions.
        
        Args:
            X (numpy.ndarray): Feature matrix
            
        Returns:
            numpy.ndarray: Normalized features
        """
        n_samples, n_features = X.shape
        
        if self.feature_means is None or self.feature_stds is None:
            # If not yet computed, calculate them manually (training phase)
            self.feature_means = np.zeros(n_features)
            self.feature_stds = np.zeros(n_features)
            
            # Calculate means manually
            for j in range(n_features):
                feature_sum = 0.0
                for i in range(n_samples):
                    feature_sum += X[i, j]
                self.feature_means[j] = feature_sum / n_samples
            
            # Calculate standard deviations manually
            for j in range(n_features):
                variance_sum = 0.0
                for i in range(n_samples):
                    variance_sum += (X[i, j] - self.feature_means[j]) ** 2
                variance = variance_sum / n_samples
                self.feature_stds[j] = np.sqrt(variance)  # Still using numpy for square root
        
        # Handle zero standard deviation (constant features) manually
        safe_stds = np.zeros_like(self.feature_stds)
        for j in range(len(self.feature_stds)):
            safe_stds[j] = 1.0 if self.feature_stds[j] == 0 else self.feature_stds[j]
        
        # Normalize manually
        X_norm = np.zeros_like(X)
        for i in range(n_samples):
            for j in range(n_features):
                X_norm[i, j] = (X[i, j] - self.feature_means[j]) / safe_stds[j]
                
        return X_norm
    
    def train(self, X, y):
        """
        Train the SVM classifier using gradient descent.
        
        Args:
            X (numpy.ndarray): Training features, shape (n_samples, n_features)
            y (numpy.ndarray): Training labels, should be -1 or 1
            
        Returns:
            self: Trained classifier
        """
        # Normalize features
        X_norm = self.normalize_features(X)
        
        # Initialize weights
        n_samples, n_features = X_norm.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []
        
        # Gradient descent optimization
        for epoch in range(self.epochs):
            # Compute margin values
            margins = y * (np.dot(X_norm, self.w) + self.b)
            
            # Find misclassified points
            misclassified = margins < 1
            
            # Compute gradients
            dw = (self.w / self.regularization) - np.sum(
                y[misclassified].reshape(-1, 1) * X_norm[misclassified], axis=0
            )
            db = -np.sum(y[misclassified])
            
            # Update weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Calculate loss for monitoring
            loss = (0.5 / self.regularization) * np.dot(self.w, self.w) + \
                   np.sum(np.maximum(0, 1 - margins)) / n_samples
            
            self.loss_history.append(loss)
            
            # Optional: print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")
        
        # Identify support vectors (points near the margin)
        margins = y * (np.dot(X_norm, self.w) + self.b)
        # Support vectors are points close to the margin (within some epsilon)
        epsilon = 1e-3
        sv_indices = np.where(np.abs(margins - 1.0) < epsilon)[0]
        self.support_vectors = X_norm[sv_indices]
        
        return self
        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X (numpy.ndarray): Samples, shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted class labels (-1 or 1)
        """
        # Normalize features
        X_norm = self.normalize_features(X)
        
        # Compute decision function
        decision = np.dot(X_norm, self.w) + self.b
        
        # Convert to class labels
        y_pred = np.sign(decision)
        
        return y_pred
        
    def decision_function(self, X):
        """
        Compute decision function (distance to hyperplane).
        
        Args:
            X (numpy.ndarray): Samples, shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Decision function values
        """
        # Normalize features
        X_norm = self.normalize_features(X)
        
        # Compute distance to hyperplane
        return np.dot(X_norm, self.w) + self.b

    def plot_loss_curve(self):
        """
        Plot training loss vs. epoch.
        
        Returns:
            str: Path to saved loss curve plot
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, 'b-')
        plt.title('SVM Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        os.makedirs(SVM_DIR, exist_ok=True)
        filename = get_timestamped_filename("loss_curve", "png")
        output_path = os.path.join(SVM_DIR, filename)
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
        
    def plot_decision_boundary(self, X, y):
        """
        Plot decision boundary on 2D feature space.
        
        Args:
            X (numpy.ndarray): Features, shape (n_samples, 2)
            y (numpy.ndarray): Labels, shape (n_samples,)
            
        Returns:
            str: Path to saved decision boundary plot
        """
        # Only works for 2D features
        if X.shape[1] != 2:
            print("Warning: Decision boundary can only be plotted for 2D features")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Normalize features
        X_norm = self.normalize_features(X)
        
        # Create a mesh grid
        h = .02  # step size in the mesh
        x_min, x_max = X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1
        y_min, y_max = X_norm[:, 1].min() - 1, X_norm[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Compute decision function on the mesh grid
        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], self.w) + self.b
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'red'],
                   linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
        
        # Fill the regions
        plt.contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')],
                    colors=['#FFAAAA', '#AAFFAA'], alpha=0.3)
        
        # Plot the training data
        plt.scatter(X_norm[y == -1][:, 0], X_norm[y == -1][:, 1], 
                   c='red', label='apple scab', s=50, alpha=0.7)
        plt.scatter(X_norm[y == 1][:, 0], X_norm[y == 1][:, 1], 
                   c='green', label='healthy', s=50, alpha=0.7)
        
        # Highlight support vectors if available
        if self.support_vectors is not None and len(self.support_vectors) > 0:
            plt.scatter(self.support_vectors[:, 0], self.support_vectors[:, 1], 
                       s=100, linewidth=1, facecolors='none', 
                       edgecolors='black', label='Support Vectors')
        
        # Add labels and legend
        plt.xlabel('Normalized Mean Saturation')
        plt.ylabel('Normalized Black Spot Ratio')
        plt.title('SVM Decision Boundary')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        os.makedirs(SVM_DIR, exist_ok=True)
        filename = get_timestamped_filename("decision_boundary", "png")
        output_path = os.path.join(SVM_DIR, filename)
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
