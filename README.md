# Telecom Churn Prediction Project

A comprehensive machine learning project for predicting customer churn in the telecommunications industry.

## 📋 Project Overview

Customer churn is a critical metric for telecom companies. This project implements a complete machine learning pipeline to predict which customers are likely to churn, enabling proactive retention strategies.

### Key Features

- **Data Preprocessing**: Automated data cleaning, encoding, and scaling
- **Multiple ML Models**: Comparison of 7 different algorithms
- **Visualization**: Comprehensive EDA and results visualization
- **Production Ready**: Modular code structure with training and prediction scripts
- **Sample Data Generator**: Built-in synthetic data generator for testing

## 🏗️ Project Structure

```
Project3/
├── data/
│   ├── raw/                    # Raw data files
│   │   └── .gitkeep
│   └── processed/              # Processed data files
│       └── .gitkeep
├── models/                     # Saved model files
│   └── .gitkeep
├── notebooks/
│   └── EDA_Telecom_Churn.ipynb # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data preprocessing module
│   ├── model_training.py       # Model training module
│   ├── visualization.py        # Visualization utilities
│   └── data_generator.py       # Sample data generator
├── train.py                    # Main training script
├── predict.py                  # Prediction script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/arpanbhandari5/Project3.git
cd Project3
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### 1. Generate Sample Data (Optional)

If you don't have a dataset, generate sample data:

```bash
python -c "from src.data_generator import generate_sample_telecom_data; generate_sample_telecom_data(5000, 'data/raw/telecom_churn_data.csv')"
```

### 2. Train Models

Run the training pipeline to train and evaluate multiple models:

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train 7 different machine learning models
- Compare model performance
- Save the best model to `models/best_churn_model.pkl`
- Generate model comparison results

### 3. Make Predictions

Use the trained model to predict churn on new data:

```bash
python predict.py --data data/raw/telecom_churn_data.csv --model models/best_churn_model.pkl
```

### 4. Exploratory Data Analysis

Open the Jupyter notebook for detailed EDA:

```bash
jupyter notebook notebooks/EDA_Telecom_Churn.ipynb
```

## 🤖 Models Implemented

The project implements and compares the following models:

1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Non-linear tree-based model
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential ensemble method
5. **XGBoost** - Optimized gradient boosting
6. **Naive Bayes** - Probabilistic classifier
7. **Support Vector Machine (SVM)** - Kernel-based classifier

## 📈 Model Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## 🔍 Dataset Features

The dataset includes the following features:

### Customer Demographics
- Gender, Senior Citizen status
- Partner, Dependents

### Services
- Phone Service, Multiple Lines
- Internet Service, Online Security
- Online Backup, Device Protection
- Tech Support, Streaming TV, Streaming Movies

### Account Information
- Tenure (months)
- Contract type
- Payment method
- Paperless billing

### Charges
- Monthly Charges
- Total Charges

### Target Variable
- **Churn**: Yes/No (whether customer left)

## 📊 Sample Results

Typical model performance (on generated data):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 0.85 | 0.82 | 0.78 | 0.80 |
| Random Forest | 0.84 | 0.81 | 0.76 | 0.78 |
| Gradient Boosting | 0.83 | 0.80 | 0.75 | 0.77 |

## 🛠️ Technology Stack

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter** - Interactive notebooks

## 📝 Code Modules

### data_preprocessing.py
Contains the `DataPreprocessor` class for:
- Loading data
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Train-test splitting

### model_training.py
Contains the `ChurnModelTrainer` class for:
- Training multiple models
- Model evaluation
- Model comparison
- Saving/loading models
- Making predictions

### visualization.py
Contains the `ChurnVisualizer` class for:
- Feature distribution plots
- Correlation matrices
- Confusion matrices
- ROC curves
- Feature importance
- Model comparison plots

### data_generator.py
Generates synthetic telecom churn data for testing and demonstration.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

- Arpan Bhandari

## 🙏 Acknowledgments

- Dataset features inspired by IBM Telco Customer Churn dataset
- Built with scikit-learn and modern ML best practices

## 📧 Contact

For questions or feedback, please open an issue in the repository.
