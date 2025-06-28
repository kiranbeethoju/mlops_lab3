# Handwritten Digit Recognition

This project demonstrates how to use scikit-learn to recognize handwritten digits. It trains a Support Vector Machine (SVM) classifier on the digits dataset and evaluates its performance.

## Refactoring

The original code from the scikit-learn example was refactored to:
- Remove redundant code that was generating the classification report twice.
- Improve the plotting logic by creating a reusable function to display digits, which cleaned up the code and removed duplication.

## How to Run

1.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the script:
    ```bash
    python plot_digits_classfication.py
    ```

This will train the model, print a classification report and a confusion matrix to the console, and display plots showing some of the training digits and the predictions on the test set. 