"""
Sample Data Generator for Telecom Churn Prediction
"""

import pandas as pd
import numpy as np
import os


def generate_sample_telecom_data(n_samples=5000, output_path=None):
    """
    Generate sample telecom churn dataset
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the CSV file
        
    Returns:
        pd.DataFrame: Generated dataset
    """
    np.random.seed(42)
    
    # Customer demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    
    # Account information
    tenure = np.random.randint(1, 73, n_samples)  # months
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1])
    multiple_lines = np.where(
        phone_service == 'Yes',
        np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.5, 0.4, 0.1]),
        'No phone service'
    )
    
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.45, 0.15])
    
    online_security = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'No internet service'
    )
    
    online_backup = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'No internet service'
    )
    
    device_protection = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'No internet service'
    )
    
    tech_support = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'No internet service'
    )
    
    streaming_tv = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
        'No internet service'
    )
    
    streaming_movies = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
        'No internet service'
    )
    
    # Billing information
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                n_samples, p=[0.55, 0.2, 0.25])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ], n_samples, p=[0.35, 0.2, 0.25, 0.2])
    
    # Charges
    monthly_charges = np.random.uniform(18.0, 120.0, n_samples)
    total_charges = monthly_charges * tenure + np.random.normal(0, 50, n_samples)
    total_charges = np.maximum(total_charges, monthly_charges)  # Ensure total >= monthly
    
    # Churn (target variable)
    # Higher churn probability for:
    # - Month-to-month contracts
    # - High monthly charges
    # - Low tenure
    # - Senior citizens
    # - No tech support/online security
    
    churn_prob = 0.2  # Base probability
    
    churn_prob_array = np.full(n_samples, churn_prob)
    
    # Adjust based on features
    churn_prob_array = np.where(contract == 'Month-to-month', churn_prob_array + 0.2, churn_prob_array)
    churn_prob_array = np.where(contract == 'Two year', churn_prob_array - 0.15, churn_prob_array)
    churn_prob_array = np.where(tenure < 12, churn_prob_array + 0.15, churn_prob_array)
    churn_prob_array = np.where(tenure > 48, churn_prob_array - 0.1, churn_prob_array)
    churn_prob_array = np.where(senior_citizen == 1, churn_prob_array + 0.05, churn_prob_array)
    churn_prob_array = np.where(tech_support == 'No', churn_prob_array + 0.05, churn_prob_array)
    churn_prob_array = np.where(online_security == 'No', churn_prob_array + 0.05, churn_prob_array)
    churn_prob_array = np.where(monthly_charges > 80, churn_prob_array + 0.1, churn_prob_array)
    churn_prob_array = np.where(payment_method == 'Electronic check', churn_prob_array + 0.05, churn_prob_array)
    
    # Clip probabilities to [0, 1]
    churn_prob_array = np.clip(churn_prob_array, 0, 1)
    
    churn = np.random.binomial(1, churn_prob_array)
    churn_label = np.where(churn == 1, 'Yes', 'No')
    
    # Create dataframe
    df = pd.DataFrame({
        'customerID': [f'CUST{i:05d}' for i in range(n_samples)],
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'Churn': churn_label
    })
    
    # Save to CSV if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Churn rate: {(churn.sum() / len(churn) * 100):.2f}%")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_file = os.path.join(project_root, "data", "raw", "telecom_churn_data.csv")
    df = generate_sample_telecom_data(n_samples=5000, output_path=output_file)
    
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nChurn distribution:")
    print(df['Churn'].value_counts())
