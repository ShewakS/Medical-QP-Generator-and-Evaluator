# MongoDB Setup Guide

## 🗄️ Database Migration Complete!

Your project has been successfully migrated from SQLite/MySQL to **MongoDB**.

---

## 📋 Prerequisites

### 1. Install MongoDB

**Windows:**
1. Download MongoDB Community Server from: https://www.mongodb.com/try/download/community
2. Run the installer (choose "Complete" installation)
3. Install MongoDB as a Windows Service (recommended)
4. MongoDB will start automatically

**Alternative - Using MongoDB Atlas (Cloud):**
- Sign up at: https://www.mongodb.com/cloud/atlas
- Create a free cluster
- Get your connection string

---

## 🚀 Quick Start

### Step 1: Install Python Packages
```bash
pip install -r requirements.txt
```

This installs:
- `flask-mongoengine` - MongoDB integration for Flask
- `mongoengine` - Python ODM for MongoDB
- `pymongo` - MongoDB driver

### Step 2: Start MongoDB (Local Installation)

**Windows:**
MongoDB should start automatically if installed as a service. To check:
```bash
mongosh
```

If not running, start it with:
```bash
net start MongoDB
```

### Step 3: Initialize Database
```bash
python setup.py
```

This will:
- Connect to MongoDB
- Create test user (username: `testuser`, password: `test123`)
- Configure collections (created automatically)

### Step 4: Run Application
```bash
python run_app.py
```

Then open: **http://localhost:5000**

---

## ⚙️ Configuration

### Local MongoDB (Default)
By default, the app connects to:
- **Host:** localhost
- **Port:** 27017
- **Database:** medquiz

No additional configuration needed!

### Custom MongoDB Configuration

Create a `.env` file in your project root:

```env
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DB=medquiz
MONGODB_USERNAME=
MONGODB_PASSWORD=
SECRET_KEY=your-secret-key-here
```

### MongoDB Atlas (Cloud) Configuration

```env
MONGODB_HOST=cluster0.xxxxx.mongodb.net
MONGODB_PORT=27017
MONGODB_DB=medquiz
MONGODB_USERNAME=your_username
MONGODB_PASSWORD=your_password
SECRET_KEY=your-secret-key-here
```

---

## 📊 Database Structure

### Collections

#### `users`
- `username` (String, unique)
- `email` (String, unique)
- `password_hash` (String)
- `created_at` (DateTime)
- `last_login` (DateTime)
- `is_active` (Boolean)

#### `test_results`
- `user` (Reference to User)
- `topic` (String)
- `num_questions` (Integer)
- `score` (Float)
- `total_questions` (Integer)
- `answers` (Dictionary)
- `generated_questions` (List)
- `created_at` (DateTime)
- `updated_at` (DateTime)

---

## 🔍 Viewing Your Data

### Using MongoDB Compass (GUI)
1. Download from: https://www.mongodb.com/try/download/compass
2. Connect to: `mongodb://localhost:27017`
3. Browse the `medquiz` database

### Using MongoDB Shell
```bash
mongosh
use medquiz
db.users.find()
db.test_results.find()
```

---

## 🛠️ Common Commands

### View all users
```bash
mongosh
use medquiz
db.users.find().pretty()
```

### View all test results
```bash
db.test_results.find().pretty()
```

### Delete all test results
```bash
db.test_results.deleteMany({})
```

### Drop entire database
```bash
db.dropDatabase()
```

---

## 🔧 Troubleshooting

### "Connection refused" error
- Make sure MongoDB is running: `mongosh`
- Start MongoDB service: `net start MongoDB`
- Check if port 27017 is available

### "Authentication failed" error
- Check username/password in `.env` file
- Verify credentials in MongoDB

### Can't connect to MongoDB Atlas
- Check your IP is whitelisted in Atlas
- Verify connection string format
- Ensure username/password are correct

---

## ✨ What Changed?

### Replaced:
- ❌ SQLite/MySQL → ✅ MongoDB
- ❌ Flask-SQLAlchemy → ✅ Flask-MongoEngine
- ❌ SQL queries → ✅ MongoDB queries

### Benefits:
- 🚀 **Flexible Schema** - Easy to add new fields
- 📊 **Better JSON Support** - Native document storage
- ⚡ **Scalability** - Better for growing applications
- 🌐 **Cloud Ready** - Easy to use with MongoDB Atlas

---

## 📚 Additional Resources

- MongoDB Documentation: https://docs.mongodb.com/
- MongoEngine Documentation: http://mongoengine.org/
- Flask-MongoEngine: https://github.com/mongoengine/flask-mongoengine
- MongoDB University (Free Courses): https://university.mongodb.com/

---

Need help? Check the [QUICKSTART.md](QUICKSTART.md) for general app usage.
