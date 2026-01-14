#!/usr/bin/env python3
"""
Setup script to initialize the application with authentication and database
"""

import os
import sys
from pathlib import Path

def setup_database():
    """Initialize the database"""
    print("\n" + "="*60)
    print("Setting up Database...")
    print("="*60)
    
    try:
        # Import after packages are checked
        from api import app
        
        with app.app_context():
            # MongoDB doesn't need table creation like SQL databases
            # Collections are created automatically when first document is inserted
            print("✓ MongoDB connection configured successfully!")
            print("  - Database: medquiz")
            print("  - Collections will be created automatically")
            
            # Show database info
            mongodb_settings = app.config.get('MONGODB_SETTINGS', {})
            db_name = mongodb_settings.get('db', 'medquiz')
            host = mongodb_settings.get('host', 'localhost')
            port = mongodb_settings.get('port', 27017)
            
            print(f"\n✓ MongoDB Connection: {host}:{port}/{db_name}")
            print("  Note: Make sure MongoDB is running on your system")
            
    except Exception as e:
        print(f"✗ Error setting up database: {e}")
        print("\n  Make sure MongoDB is installed and running:")
        print("  - Download from: https://www.mongodb.com/try/download/community")
        print("  - Or run: mongod")
        return False
    
    return True

def check_requirements():
    """Check if all required packages are installed"""
    print("\n" + "="*60)
    print("Checking Requirements...")
    print("="*60)
    
    required_packages = {
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'mongoengine': 'MongoEngine',
        'pymongo': 'PyMongo',
        'flask_login': 'Flask-Login',
        'werkzeug': 'Werkzeug',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-Learn',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n✗ Missing packages: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed!")
    return True

def create_test_user():
    """Create a test user for demonstration"""
    print("\n" + "="*60)
    print("Creating Test User...")
    print("="*60)
    
    try:
        from api import app
        from models import User
        
        with app.app_context():
            # Check if test user already exists
            test_user = User.objects(username='testuser').first()
            
            if test_user:
                print("✓ Test user already exists!")
                print(f"  Username: testuser")
                print(f"  Password: test123")
            else:
                # Create test user
                new_user = User(username='testuser', email='test@example.com')
                new_user.set_password('test123')
                new_user.save()
                print("✓ Test user created successfully!")
                print(f"  Username: testuser")
                print(f"  Password: test123")
            
            print("\n⚠ IMPORTANT: Change this password after first login!")
        
    except Exception as e:
        print(f"✗ Error creating test user: {e}")
        return False
    
    return True

def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " SmartAI MedQuiz - Setup Script ".center(58) + "║")
    print("║" + " Authentication & Database Setup ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n" + "!"*60)
        print("Please install missing packages and try again.")
        print("!"*60)
        sys.exit(1)
    
    # Step 2: Setup database
    if not setup_database():
        print("\n" + "!"*60)
        print("Database setup failed. Please check the error above.")
        print("!"*60)
        sys.exit(1)
    
    # Step 3: Create test user
    create_test_user()
    
    # Final message
    print("\n" + "="*60)
    print("Setup Complete! 🎉")
    print("="*60)
    print("\nNext Steps:")
    print("1. Start the application:")
    print("   python run_app.py")
    print("\n2. Open your browser:")
    print("   http://localhost:5000")
    print("\n3. Login with test credentials:")
    print("   Username: testuser")
    print("   Password: test123")
    print("\n4. (Optional) Create your own account")
    print("\nFor more information, see AUTHENTICATION_SETUP.md")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
