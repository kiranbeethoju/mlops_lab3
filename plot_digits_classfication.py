"""
Recognizing hand-written digits using scikit-learn with hyperparameter optimization
"""

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

def plot_some_digits(images, labels, title_prefix, n_to_plot=4):
    _, axes = plt.subplots(nrows=1, ncols=n_to_plot, figsize=(10, 3))
    for ax, image, label in zip(axes, images, labels):
        ax.set_axis_off()
        if image.ndim == 1:
            image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"{title_prefix}: {label}")

# Load digits dataset
digits = datasets.load_digits()
plot_some_digits(digits.images, digits.target, "Training")

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Hyperparameter optimization for test_size (dev_size)
print("Optimizing test_size (dev_size)...")
test_sizes = [0.2, 0.3, 0.4, 0.5]
best_score = 0
best_test_size = 0.5

for test_size in test_sizes:
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
        data, digits.target, test_size=test_size, shuffle=False, random_state=42
    )
    
    # Quick evaluation with default parameters
    clf_temp = svm.SVC(gamma=0.001, C=1.0)
    clf_temp.fit(X_train_temp, y_train_temp)
    score = clf_temp.score(X_test_temp, y_test_temp)
    
    print(f"Test size: {test_size}, Accuracy: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_test_size = test_size

print(f"Best test_size: {best_test_size} with accuracy: {best_score:.4f}")

# Split data with optimal test_size
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=best_test_size, shuffle=False, random_state=42
)

# Hyperparameter optimization for gamma and C
print("\nOptimizing gamma and C parameters...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
}

# Create SVM classifier
svm_classifier = svm.SVC()

# Perform grid search
grid_search = GridSearchCV(
    svm_classifier, 
    param_grid, 
    cv=3, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1
)

print("Performing grid search...")
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\nBest parameters found:")
print(f"C: {best_params['C']}")
print(f"gamma: {best_params['gamma']}")
print(f"Best cross-validation score: {best_score:.4f}")

# Train final model with best parameters
clf = grid_search.best_estimator_
predicted = clf.predict(X_test)

# Visualize predictions
plot_some_digits(X_test, predicted, "Prediction")

# Print classification report
final_accuracy = clf.score(X_test, y_test)
print(f"\nFinal test accuracy with optimized parameters: {final_accuracy:.4f}")
print(
    f"Classification report for optimized classifier:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# Display confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix - Optimized Model")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

# Summary of optimization results
print(f"\n{'='*50}")
print("HYPERPARAMETER OPTIMIZATION SUMMARY")
print(f"{'='*50}")
print(f"Optimal test_size: {best_test_size}")
print(f"Optimal C: {best_params['C']}")
print(f"Optimal gamma: {best_params['gamma']}")
print(f"Final test accuracy: {final_accuracy:.4f}")
print(f"{'='*50}")

plt.show() 