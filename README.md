# Handwritten Digit Recognition with Hyperparameter Optimization

This project demonstrates how to use scikit-learn to recognize handwritten digits. It trains a Support Vector Machine (SVM) classifier on the digits dataset and evaluates its performance with optimal hyperparameter selection.

## Features

### Hyperparameter Optimization
The project includes comprehensive hyperparameter optimization using GridSearchCV to find the best combination of:
- **dev_size**: Test size for train-test split (0.2, 0.3, 0.4, 0.5)
- **gamma**: SVM gamma parameter (0.001, 0.01, 0.1, 1)
- **C**: SVM regularization parameter (1, 10, 100, 1000)

The system automatically:
1. Tests different combinations of hyperparameters using 3-fold cross-validation
2. Selects the best performing combination based on validation accuracy
3. Trains the final model using the optimal parameters
4. Reports the best parameters and performance metrics

### Code Structure
- Clean, modular code with reusable plotting functions
- Comprehensive evaluation with classification reports and confusion matrices
- Automated hyperparameter tuning for optimal performance

## How to Run

1. Install the dependencies:
   ```bash
   pip install matplotlib scikit-learn
   ```

2. Run the script:
   ```bash
   python plot_digits_classfication.py
   ```

The script will automatically:
- Find the optimal hyperparameters through grid search
- Display the best parameter combination and cross-validation accuracy
- Show classification report with precision, recall, and f1-scores
- Generate confusion matrix visualization
- Display sample predictions with visualizations

## Continuous Integration

The project includes GitHub Actions workflow (`demoactions.yml`) that automatically:
- Runs on every push to any branch
- Installs dependencies from requirements.txt
- Executes the digit recognition script
- Validates the hyperparameter optimization process

## Recent Updates

- **Hyperparameter Optimization**: Added GridSearchCV for automatic parameter tuning
- **Performance Metrics**: Enhanced reporting with best parameter tracking
- **Code Optimization**: Improved efficiency with systematic parameter search
- **Documentation**: Comprehensive README with usage instructions and feature descriptions

## Requirements

- Python 3.10+
- matplotlib
- scikit-learn

## Repository Structure

```
├── plot_digits_classfication.py    # Main script with hyperparameter optimization
├── .github/workflows/demoactions.yml  # CI/CD workflow
├── mlops_lab3/requirements.txt     # Python dependencies
└── README.md                       # This documentation
``` 