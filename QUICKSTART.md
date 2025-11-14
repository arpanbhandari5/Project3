# Quick Start Guide

Get started with the Telecom Churn Prediction project in minutes!

## 🚀 Quick Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Data (Optional)

If you don't have your own dataset:

```bash
python src/data_generator.py
```

This creates a sample dataset at `data/raw/telecom_churn_data.csv`

### Step 3: Train Models

```bash
python train.py
```

Expected output:
- Trains 7 different ML models
- Compares their performance
- Saves the best model to `models/best_churn_model.pkl`
- Takes approximately 1-2 minutes

### Step 4: Make Predictions

```bash
python predict.py
```

This will:
- Load the trained model
- Make predictions on the dataset
- Save results to `data/raw/telecom_churn_data_predictions.csv`

## 📊 View Analysis

### Jupyter Notebook

```bash
jupyter notebook notebooks/EDA_Telecom_Churn.ipynb
```

Explore:
- Data distribution
- Feature correlations
- Churn patterns
- Visual insights

## 🎯 Using Your Own Data

### Data Format

Your CSV file should include these columns:

**Required:**
- `Churn` - Target variable (Yes/No)

**Features (examples):**
- Customer demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Services: `PhoneService`, `InternetService`, `OnlineSecurity`, etc.
- Account info: `tenure`, `Contract`, `PaymentMethod`
- Charges: `MonthlyCharges`, `TotalCharges`

### Train on Your Data

```bash
# Modify train.py to point to your data file
python train.py
```

Or edit the `DATA_PATH` variable in `train.py`:
```python
DATA_PATH = "path/to/your/data.csv"
```

## 📁 Project Structure

```
Project3/
├── data/
│   ├── raw/              # Your data files
│   └── processed/        # Processed data
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code modules
├── train.py              # Training script
└── predict.py            # Prediction script
```

## 🔍 Model Performance

After training, check:

1. **Console Output** - Shows metrics for all models
2. **models/model_comparison_results.csv** - Detailed comparison
3. **models/best_churn_model.pkl** - Best performing model

## 🛠️ Troubleshooting

### Issue: Module not found error
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Data file not found
**Solution:** Generate sample data or specify correct path
```bash
python src/data_generator.py
```

### Issue: Memory error during training
**Solution:** Reduce dataset size or use fewer models

## 📈 Next Steps

1. **Tune Hyperparameters** - Edit `config.py` to adjust model parameters
2. **Feature Engineering** - Modify preprocessing in `src/data_preprocessing.py`
3. **Add New Models** - Extend `src/model_training.py`
4. **Create Visualizations** - Use `src/visualization.py` utilities

## 💡 Tips

- Start with the default settings
- Check model comparison results to understand performance
- Use the Jupyter notebook for detailed exploration
- Adjust preprocessing based on your data characteristics

## 📞 Need Help?

Check the main [README.md](README.md) for detailed documentation.

---

**Happy Modeling! 🎉**
