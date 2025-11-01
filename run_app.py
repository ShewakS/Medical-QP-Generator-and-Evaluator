#!/usr/bin/env python3
import sys
from pathlib import Path

def check_requirements():
    required_packages = ['flask', 'flask_cors', 'pandas', 'numpy', 'sklearn']
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    if missing_packages:
        print(" Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n Install missing packages with:")
        print("   pip install -r requirements_api.txt")
        return False
    return True

def check_models():
    models_dir = Path('trained_models')
    required_models = ['voting_ensemble_model.pkl', 'preprocessing_components.pkl']
    missing_models = []
    for model in required_models:
        if not (models_dir / model).exists():
            missing_models.append(model)
    if missing_models:
        print("  Some trained models are missing:")
        for model in missing_models:
            print(f"   - {model}")
        print("\n Run the ML pipeline first:")
        print("   python ml_pipeline.py")
        return False
    return True

def main():
    print(" Medical Question Generator & Evaluator")
    print("="*50)
    print(" Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print(" All packages installed")
    print(" Checking trained models...")
    if not check_models():
        print("  Models missing, but app will work with mock data")
    else:
        print(" All models found")
    print("\n Starting web application...")
    try:
        from api import app
        print(" Open your browser and go to: http://localhost:5000")
        print(" Press Ctrl+C to stop the server")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n Server stopped by user")
    except Exception as e:
        print(f" Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
