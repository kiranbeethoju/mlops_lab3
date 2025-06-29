"""
Recognizing hand-written digits using scikit-learn
"""

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV

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

# Hyperparameter optimization
param_grid = {'gamma': [0.001, 0.01, 0.1, 1], 'C': [1, 10, 100, 1000]}
best_dev_acc = 0
best_params = {}
best_clf = None

# Try different dev_size values
for dev_size in [0.2, 0.3, 0.4, 0.5]:
    # Split data with current dev_size
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=dev_size, shuffle=False
    )
    
    # Create and train GridSearchCV
    clf = GridSearchCV(svm.SVC(), param_grid, cv=3)
    clf.fit(X_train, y_train)
    
    # Check if this is the best configuration
    dev_acc = clf.best_score_
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_params = {'dev_size': dev_size, **clf.best_params_}
        best_clf = clf.best_estimator_
        # Keep the best split for final evaluation
        X_train_best, X_test_best, y_train_best, y_test_best = X_train, X_test, y_train, y_test

print(f"Best parameters: {best_params}")
print(f"Best cross-validation accuracy: {best_dev_acc:.4f}")

# Train final model with best parameters
predicted = best_clf.predict(X_test_best)

# Visualize predictions
plot_some_digits(X_test_best, predicted, "Prediction")

# Print classification report
print(
    f"Classification report for classifier {best_clf}:\n"
    f"{metrics.classification_report(y_test_best, predicted)}\n"
)

# Display confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test_best, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show() 