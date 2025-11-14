"""
Setup script for Telecom Churn Prediction Project
"""

import os
import sys


def create_directories():
    """
    Create necessary directories for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'notebooks',
        'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create .gitkeep files
    gitkeep_dirs = ['data/raw', 'data/processed', 'models']
    for directory in gitkeep_dirs:
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, 'w').close()
            print(f"✓ Created .gitkeep in: {directory}")


def generate_sample_data():
    """
    Generate sample data for testing
    """
    try:
        sys.path.append('src')
        from data_generator import generate_sample_telecom_data
        
        output_path = 'data/raw/telecom_churn_data.csv'
        
        if os.path.exists(output_path):
            response = input(f"\n{output_path} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Skipping data generation.")
                return
        
        print("\nGenerating sample telecom churn data...")
        generate_sample_telecom_data(n_samples=5000, output_path=output_path)
        print(f"✓ Sample data generated successfully at: {output_path}")
        
    except Exception as e:
        print(f"✗ Error generating sample data: {e}")


def check_dependencies():
    """
    Check if required packages are installed
    """
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 
        'seaborn', 'xgboost', 'joblib', 'jupyter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed")
        return True


def main():
    """
    Main setup function
    """
    print("=" * 70)
    print("TELECOM CHURN PREDICTION PROJECT - SETUP")
    print("=" * 70)
    
    print("\n1. Creating project directories...")
    create_directories()
    
    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()
    
    if deps_ok:
        print("\n3. Sample data generation...")
        response = input("Generate sample data? (y/n): ")
        if response.lower() == 'y':
            generate_sample_data()
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    
    if deps_ok:
        print("\nNext steps:")
        print("1. Run 'python train.py' to train models")
        print("2. Run 'python predict.py' to make predictions")
        print("3. Open 'notebooks/EDA_Telecom_Churn.ipynb' for exploratory analysis")
    else:
        print("\nPlease install dependencies first:")
        print("pip install -r requirements.txt")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
