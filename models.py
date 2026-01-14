from mongoengine import Document, StringField, DateTimeField, BooleanField, ReferenceField, IntField, FloatField, DictField, ListField, connect
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# MongoDB connection will be initialized in api.py

class User(UserMixin, Document):
    meta = {'collection': 'users'}
    
    username = StringField(required=True, unique=True, max_length=80)
    name = StringField(max_length=150)  # User's display name
    email = StringField(required=True, unique=True, max_length=120)
    password_hash = StringField(required=True, max_length=255)
    created_at = DateTimeField(default=datetime.utcnow)
    last_login = DateTimeField()
    is_active = BooleanField(default=True)
    
    def set_password(self, password):
        """Hash and set the password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches the hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_id(self):
        """Return the user ID as string for Flask-Login"""
        return str(self.id)
    
    def get_test_results(self):
        """Get all test results for this user"""
        return TestResult.objects(user=self).order_by('-created_at')
    
    def __repr__(self):
        return f'<User {self.username}>'


class TestResult(Document):
    meta = {'collection': 'test_results'}
    
    user = ReferenceField(User, required=True)
    topic = StringField(required=True, max_length=100)
    num_questions = IntField(required=True)
    score = FloatField()
    total_questions = IntField()
    answers = DictField()  # Store answers as dictionary
    generated_questions = ListField()  # Store generated questions as list
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super(TestResult, self).save(*args, **kwargs)
    
    def __repr__(self):
        return f'<TestResult user={self.user.username} topic={self.topic}>'
