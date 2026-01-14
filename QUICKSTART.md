# 🚀 Quick Start Guide - Authentication & Database

## What You Need to Know (2-Minute Read)

### ✨ New Features Added
1. **Login/Register System** - Create accounts and login
2. **Secure Database** - User data stored safely with encrypted passwords
3. **Test History** - Your test scores are saved automatically
4. **User Sessions** - Stay logged in across page refreshes

---

## 🎯 Getting Started in 3 Steps

### Step 1: Install Packages
```bash
pip install -r requirements.txt
```

### Step 2: Initialize Database
```bash
python setup.py
```

This creates:
- ✓ Database file (`medquiz.db`)
- ✓ User and test tables
- ✓ Test user account (testuser / test123)

### Step 3: Run Application
```bash
python run_app.py
```

Then open: **http://localhost:5000**

---

## 📱 Using the App

### First Time Users - Register
1. Click "Register here"
2. Create username (3+ chars), email, password (6+ chars)
3. Click "Create Account"
4. You're automatically logged in!

### Returning Users - Login
1. Enter your username and password
2. Click "Sign In"
3. Access your dashboard

### Using Dashboard
- Same as before! Generate questions, take tests
- **Your results are automatically saved to database**
- See your test history anytime

### Logout
- Click "Logout" button (top right)
- Redirected back to login page

---

## 📊 Database Info

**File**: `medquiz.db` (automatically created)

**Tables**:
- `users` - Your account info (username, email, hashed password)
- `test_results` - Your test scores and answers

**Data Saved**:
- Test score (percentage)
- Medical topic
- Correct/incorrect answers
- Time taken
- Date of test

---

## 🔑 Login Credentials

**Test Account** (auto-created by setup.py):
```
Username: testuser
Password: test123
```

⚠️ **Change this after first login!**

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named flask_sqlalchemy" | Run: `pip install -r requirements.txt` |
| Port 5000 already in use | Edit api.py last line, change to port 5001 |
| Forgot password | Delete `medquiz.db`, run `setup.py` again |
| Database locked | Delete `medquiz.db`, restart app |

---

## 📁 New Files Created

```
📄 models.py                    - Database setup
📄 templates/login.html         - Login page
📄 templates/dashboard.html     - Main app with logout
📄 setup.py                     - Database initialization
📄 AUTHENTICATION_SETUP.md      - Full documentation
📄 SETUP_SUMMARY.md             - Detailed guide
📄 medquiz.db                   - SQLite database (created automatically)
```

---

## ⚡ Key API Changes

**Old Route**:
- `/` → Showed main page

**New Routes**:
- `/` → Redirects to login/dashboard
- `/login` → Login/register page
- `/dashboard` → Main app (login required)
- `/auth/login` → API login
- `/auth/register` → API register
- `/auth/logout` → API logout
- `/api/save-test-result` → Save test scores
- `/api/user-test-results` → Get your test history

---

## 🔒 Security

✅ Passwords are encrypted (not stored plain text)
✅ SQL injection protected
✅ Session tokens for security
✅ Form validation on both sides

---

## ❓ Common Questions

**Q: Where is my data stored?**
A: In `medquiz.db` file in your project folder

**Q: Can I use PostgreSQL instead?**
A: Yes! Edit `api.py` DATABASE_URL setting

**Q: Is my password safe?**
A: Yes! We use industry-standard encryption (PBKDF2)

**Q: Can I export my test results?**
A: Yes! Use the "Download Results" button on results page

**Q: Can multiple users use this?**
A: Yes! Each user has separate accounts and test history

---

## 📚 Documentation Files

- **SETUP_SUMMARY.md** - Complete technical documentation
- **AUTHENTICATION_SETUP.md** - Detailed setup & customization guide
- **QUICKSTART.md** - This file!

---

## 🎓 What to Do Next

1. ✅ Run `python setup.py`
2. ✅ Run `python run_app.py`
3. ✅ Test with testuser/test123 account
4. ✅ Create your own account
5. ✅ Take a test and see results saved!
6. ✅ Read AUTHENTICATION_SETUP.md for advanced features

---

## 💡 Pro Tips

- Use strong passwords (8+ characters recommended)
- Save test results PDF using "Download Results" button
- Your test history is kept indefinitely
- Database backups: copy `medquiz.db` file

---

## 🆘 Need Help?

Check these files:
- `AUTHENTICATION_SETUP.md` - Setup troubleshooting
- `SETUP_SUMMARY.md` - Technical details
- `api.py` - See authentication code

---

**You're all set! Enjoy your new authentication system! 🎉**
