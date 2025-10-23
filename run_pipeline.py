#!/usr/bin/env python3
"""
Quick start script for the ML Pipeline
This script will install dependencies and run the complete pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_datasets():
    """Check if required datasets exist"""
    required_files = ['train.csv', 'test.csv', 'validation.csv']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing dataset files: {missing_files}")
        print("Please ensure all CSV files are in the current directory.")
        return False
    
    print("✅ All dataset files found!")
    return True

def run_pipeline():
    """Run the ML pipeline"""
    print("🚀 Starting ML Pipeline...")
    try:
        from ml_pipeline import OptimizedMLPipeline
        
        # Initialize and run pipeline
        pipeline = OptimizedMLPipeline(verbose=True)
        results = pipeline.run_complete_pipeline()
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"📊 Results dashboard saved to: ml_results_dashboard.html")
        
        return results
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return None

def main():
    """Main execution function"""
    print("=" * 60)
    print("🤖 ML Pipeline Quick Start")
    print("=" * 60)
    
    # Check if datasets exist
    if not check_datasets():
        return
    
    # Install dependencies
    if not install_requirements():
        return
    
    # Run pipeline
    results = run_pipeline()
    
    if results:
        print("\n" + "=" * 60)
        print("✨ EXECUTION SUMMARY")
        print("=" * 60)
        
        best_model = max(results['results'].keys(), 
                        key=lambda k: results['results'][k]['accuracy'])
        best_accuracy = results['results'][best_model]['accuracy']
        
        print(f"🏆 Best Model: {best_model}")
        print(f"🎯 Best Accuracy: {best_accuracy:.4f}")
        print(f"⏱️  Total Time: {results['total_time']:.2f} seconds")
        print(f"📊 Dashboard: ml_results_dashboard.html")
        
        # Model comparison
        print("\n📈 Model Comparison:")
        for name, result in results['results'].items():
            print(f"   {name}: {result['accuracy']:.4f} ({result['train_time']:.2f}s)")

if __name__ == "__main__":
    main()
