from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from mongoengine import connect
from models import User, TestResult
import pickle
import pandas as pd
import random
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import io
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# MongoDB Database Configuration
MONGODB_HOST = os.environ.get('MONGODB_HOST', 'localhost')
MONGODB_PORT = int(os.environ.get('MONGODB_PORT', 27017))
MONGODB_DB = os.environ.get('MONGODB_DB', 'medquiz')
MONGODB_USERNAME = os.environ.get('MONGODB_USERNAME', '')
MONGODB_PASSWORD = os.environ.get('MONGODB_PASSWORD', '')

# Connect to MongoDB
if MONGODB_USERNAME and MONGODB_PASSWORD:
    connect(
        db=MONGODB_DB,
        host=MONGODB_HOST,
        port=MONGODB_PORT,
        username=MONGODB_USERNAME,
        password=MONGODB_PASSWORD,
        authentication_source='admin'
    )
else:
    connect(db=MONGODB_DB, host=MONGODB_HOST, port=MONGODB_PORT)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

CORS(app)

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.objects(id=user_id).first()
    except:
        return None

class MedicalQuestionAPI:
    def __init__(self):
        self.models_dir = Path('trained_models')
        self.load_models()
        if not self.load_question_dataset():
            self.load_question_templates()
        self.generated_questions_cache = {}
    
    def load_models(self):
        try:
            with open(self.models_dir / 'voting_ensemble_model.pkl', 'rb') as f:
                self.voting_model = pickle.load(f)
            with open(self.models_dir / 'preprocessing_components.pkl', 'rb') as f:
                self.preprocessing = pickle.load(f)
            print(" Models loaded successfully")
        except Exception as e:
            print(f" Error loading models: {e}")
            self.voting_model = None
            self.preprocessing = None
    
    def load_question_dataset(self):
        try:
            self.train_df = pd.read_csv('train.csv')
            self.test_df = pd.read_csv('test.csv') 
            self.validation_df = pd.read_csv('validation.csv')
            self.all_questions_df = pd.concat([self.train_df, self.test_df, self.validation_df], ignore_index=True)
            self.all_questions_df = self.all_questions_df.dropna(subset=['question', 'opa', 'opb', 'opc', 'opd'])
            self.topic_mapping = {
                'anatomy': ['Anatomy'],
                'physiology': ['Physiology'],
                'pathology': ['Pathology'],
                'pharmacology': ['Pharmacology'],
                'microbiology': ['Microbiology'],
                'biochemistry': ['Biochemistry'],
                'medicine': ['Medicine'],
                'surgery': ['Surgery'],
                'pediatrics': ['Pediatrics'],
                'gynecology': ['Gynecology'],
                'psychiatry': ['Psychiatry'],
                'radiology': ['Radiology'],
                'cardiology': ['Cardiology'],
                'neurology': ['Neurology'],
                'dermatology': ['Dermatology']
            }
            print(f" Loaded {len(self.all_questions_df)} questions from dataset")
            return True
        except Exception as e:
            print(f" Error loading question dataset: {e}")
            return False
    

    
    def generate_questions_from_dataset(self, topic, count, difficulty):
        if not hasattr(self, 'all_questions_df'):
            return []
        cache_key = f"{topic}_{difficulty}"
        topic_subjects = self.topic_mapping.get(topic, ['Medicine'])
        if 'subject_name' in self.all_questions_df.columns:
            topic_questions = self.all_questions_df[
                self.all_questions_df['subject_name'].isin(topic_subjects)
            ].copy()
        else:
            topic_questions = self.all_questions_df.copy()
        if len(topic_questions) == 0:
            # If no questions found for specific topic, try general medical subjects
            general_subjects = ['Medicine', 'Surgery', 'Pathology']
            topic_questions = self.all_questions_df[
                self.all_questions_df['subject_name'].isin(general_subjects)
            ].copy()
            if len(topic_questions) == 0:
                # Last resort: use all questions
                topic_questions = self.all_questions_df.copy()
        if cache_key not in self.generated_questions_cache:
            self.generated_questions_cache[cache_key] = set()
        available_questions = []
        for _, row in topic_questions.iterrows():
            question_id = str(row['id'])
            if question_id in self.generated_questions_cache[cache_key]:
                continue
            question_text = str(row['question'])
            
            # Ensure all options are strings and not empty
            options = [
                str(row['opa']) if pd.notna(row['opa']) and str(row['opa']).strip() else 'Option A',
                str(row['opb']) if pd.notna(row['opb']) and str(row['opb']).strip() else 'Option B', 
                str(row['opc']) if pd.notna(row['opc']) and str(row['opc']).strip() else 'Option C',
                str(row['opd']) if pd.notna(row['opd']) and str(row['opd']).strip() else 'Option D'
            ]
            
            # Ensure correct option index is valid (0-3)
            correct_option = int(row['cop']) - 1 if pd.notna(row['cop']) and 1 <= int(row['cop']) <= 4 else 0
            correct_answer = options[correct_option]
            available_questions.append({
                'id': question_id,
                'question': question_text,
                'answer': correct_answer,
                'options': options,
                'correct': correct_option,
                'explanation': str(row.get('exp', f"The correct answer is {correct_answer}.")),
                'subject': str(row.get('subject_name', topic))
            })
        if len(available_questions) < count:
            print(f" Resetting question cache for {topic}_{difficulty}")
            self.generated_questions_cache[cache_key] = set()
            available_questions = []
            for _, row in topic_questions.iterrows():
                question_id = str(row['id'])
                question_text = str(row['question'])
                options = [
                    str(row['opa']) if pd.notna(row['opa']) and str(row['opa']).strip() else 'Option A',
                    str(row['opb']) if pd.notna(row['opb']) and str(row['opb']).strip() else 'Option B', 
                    str(row['opc']) if pd.notna(row['opc']) and str(row['opc']).strip() else 'Option C',
                    str(row['opd']) if pd.notna(row['opd']) and str(row['opd']).strip() else 'Option D'
                ]
                correct_option = int(row['cop']) - 1 if pd.notna(row['cop']) and 1 <= int(row['cop']) <= 4 else 0
                correct_answer = options[correct_option]
                available_questions.append({
                    'id': question_id,
                    'question': question_text,
                    'answer': correct_answer,
                    'options': options,
                    'correct': correct_option,
                    'explanation': str(row.get('exp', f"The correct answer is {correct_answer}.")),
                    'subject': str(row.get('subject_name', topic))
                })
        random.shuffle(available_questions)
        selected_questions = available_questions[:count]
        for q in selected_questions:
            self.generated_questions_cache[cache_key].add(q['id'])
        questions = []
        for i, q_data in enumerate(selected_questions):
            question = {
                'id': i + 1,
                'question': q_data['question'],
                'options': q_data['options'],
                'correct': q_data['correct'],
                'topic': topic,
                'difficulty': difficulty,
                'explanation': q_data['explanation'],
                'subject': q_data['subject']
            }
            questions.append(question)
        print(f" Generated {len(questions)} unique questions for {topic}_{difficulty}")
        return questions
    
    def classify_questions_by_difficulty(self, questions_df, target_difficulty):
        if target_difficulty == 'easy':
            filtered = questions_df[questions_df['question'].str.len() < 200].copy()
        elif target_difficulty == 'hard':
            filtered = questions_df[questions_df['question'].str.len() > 150].copy()
        else:
            filtered = questions_df[
                (questions_df['question'].str.len() >= 100) & 
                (questions_df['question'].str.len() <= 250)
            ].copy()
        if len(filtered) < 10:
            return questions_df
        return filtered
    

    
    def generate_questions(self, topic, count, difficulty):
        return self.generate_questions_from_dataset(topic, count, difficulty)
    
    def evaluate_answers(self, questions, user_answers):
        results = {
            'total_questions': len(questions),
            'correct_answers': 0,
            'incorrect_answers': 0,
            'score_percentage': 0,
            'detailed_results': []
        }
        for i, question in enumerate(questions):
            user_answer = user_answers.get(str(i))
            is_correct = user_answer == question['correct']
            if is_correct:
                results['correct_answers'] += 1
            else:
                results['incorrect_answers'] += 1
            # Ensure we have valid options
            options = question.get('options', ['Option A', 'Option B', 'Option C', 'Option D'])
            correct_answer = options[question['correct']] if question['correct'] < len(options) else 'Answer not available'
            user_answer_text = options[user_answer] if user_answer is not None and user_answer < len(options) else 'Not answered'
            
            results['detailed_results'].append({
                'question_id': question['id'],
                'question': question['question'],
                'user_answer': user_answer_text,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'explanation': question.get('explanation', 'No explanation available')
            })
        results['score_percentage'] = round((results['correct_answers'] / results['total_questions']) * 100, 2)
        return results

# Initialize the API
question_api = MedicalQuestionAPI()

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        topic = data.get('topic')
        count = int(data.get('count', 5))
        difficulty = data.get('difficulty', 'medium')
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        if count < 1 or count > 20:
            return jsonify({'error': 'Question count must be between 1 and 20'}), 400
        questions = question_api.generate_questions(topic, count, difficulty)
        return jsonify({
            'success': True,
            'questions': questions,
            'metadata': {
                'topic': topic,
                'count': count,
                'difficulty': difficulty,
                'generated_at': pd.Timestamp.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.get_json()
        questions = data.get('questions')
        user_answers = data.get('user_answers')
        if not questions or not user_answers:
            return jsonify({'error': 'Questions and user answers are required'}), 400
        results = question_api.evaluate_answers(questions, user_answers)
        return jsonify({
            'success': True,
            'results': results,
            'evaluated_at': pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/topics', methods=['GET'])
def get_topics():
    topics = ['anatomy', 'physiology', 'pathology', 'pharmacology', 'microbiology', 
             'biochemistry', 'medicine', 'surgery', 'pediatrics', 'gynecology', 
             'psychiatry', 'radiology', 'cardiology', 'neurology', 'dermatology']
    return jsonify({'success': True, 'topics': topics})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': question_api.voting_model is not None,
        'preprocessing_loaded': question_api.preprocessing is not None
    })

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    try:
        data = request.get_json()
        topic = data.get('topic')
        difficulty = data.get('difficulty')
        if topic and difficulty:
            cache_key = f"{topic}_{difficulty}"
            if cache_key in question_api.generated_questions_cache:
                del question_api.generated_questions_cache[cache_key]
                return jsonify({
                    'success': True,
                    'message': f'Cache cleared for {topic}_{difficulty}'
                })
        else:
            question_api.generated_questions_cache = {}
            return jsonify({
                'success': True,
                'message': 'All question cache cleared'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-pdf', methods=['POST'])
def download_pdf():
    try:
        data = request.get_json()
        results = data.get('results')
        if not results:
            return jsonify({'error': 'Results data is required'}), 400
        
        student_name = results.get('studentName', 'Student')
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30, alignment=1
        )
        story.append(Paragraph("Medical Question Test Results", title_style))
        story.append(Spacer(1, 10))
        
        # Add student name prominently
        student_style = ParagraphStyle(
            'StudentName', parent=styles['Heading2'], fontSize=14, spaceAfter=20, alignment=1, textColor=colors.HexColor('#2c3e50')
        )
        story.append(Paragraph(f"<b>Student:</b> {student_name}", student_style))
        story.append(Spacer(1, 10))
        
        info_data = [
            ['Test Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Topic:', results.get('topic', 'N/A')],
            ['Difficulty:', results.get('difficulty', 'N/A')],
            ['Total Questions:', str(results.get('total', 0))],
            ['Correct Answers:', str(results.get('correct', 0))],
            ['Score:', f"{results.get('percentage', 0)}%"]
        ]
        info_table = Table(info_data, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))
        story.append(Paragraph("Question Review", styles['Heading2']))
        story.append(Spacer(1, 12))
        questions = results.get('questions', [])
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"<b>Question {i}:</b> {q['question']}", styles['Normal']))
            story.append(Spacer(1, 6))
            is_correct = q.get('isCorrect', False)
            status = "✓ Correct" if is_correct else "✗ Incorrect"
            color = "green" if is_correct else "red"
            story.append(Paragraph(f"<b>Your Answer:</b> {q.get('userAnswer', 'Not answered')}", styles['Normal']))
            story.append(Paragraph(f"<b>Correct Answer:</b> {q.get('correctAnswer', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Status:</b> <font color='{color}'>{status}</font>", styles['Normal']))
            story.append(Spacer(1, 15))
        doc.build(story)
        buffer.seek(0)
        
        # Create filename with student name
        safe_name = "".join(c for c in student_name if c.isalnum() or c in (' ', '_')).replace(' ', '_')
        filename = f'{safe_name}_medical_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ======================== AUTHENTICATION ROUTES ========================

@app.route('/auth/login', methods=['POST'])
def auth_login():
    """Handle user login"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400
        
        user = User.objects(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            user.last_login = datetime.utcnow()
            user.save()
            return jsonify({'message': 'Login successful'}), 200
        else:
            return jsonify({'message': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'message': f'Login error: {str(e)}'}), 500


@app.route('/auth/register', methods=['POST'])
def auth_register():
    """Handle user registration"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not username or not email or not password:
            return jsonify({'message': 'All fields are required'}), 400
        
        if len(username) < 3:
            return jsonify({'message': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'message': 'Password must be at least 6 characters'}), 400
        
        if User.objects(username=username).first():
            return jsonify({'message': 'Username already exists'}), 400
        
        if User.objects(email=email).first():
            return jsonify({'message': 'Email already registered'}), 400
        
        # Create new user
        new_user = User(username=username, name=name or username, email=email)
        new_user.set_password(password)
        new_user.save()
        
        # Log the user in
        login_user(new_user)
        
        return jsonify({'message': 'Registration successful'}), 201
    except Exception as e:
        return jsonify({'message': f'Registration error: {str(e)}'}), 500


@app.route('/auth/logout', methods=['POST'])
@login_required
def auth_logout():
    """Handle user logout"""
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200


@app.route('/auth/user', methods=['GET'])
@login_required
def get_user():
    """Get current user info"""
    return jsonify({
        'id': str(current_user.id),
        'username': str(current_user.username),
        'name': str(current_user.name or current_user.username),  # Fall back to username if name not set
        'email': str(current_user.email)
    }), 200


# ======================== PAGE ROUTES ========================

@app.route('/login', methods=['GET'])
def login():
    """Render login page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    """Render main dashboard (previously index)"""
    return render_template('dashboard.html')


@app.route('/', methods=['GET'])
def index():
    """Redirect to login or dashboard based on auth status"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


# ======================== TEST RESULTS STORAGE ========================

@app.route('/api/save-test-result', methods=['POST'])
@login_required
def save_test_result():
    """Save test results to database"""
    try:
        data = request.get_json()
        
        test_result = TestResult(
            user=current_user,
            topic=data.get('topic', 'Unknown'),
            num_questions=data.get('num_questions', 0),
            score=data.get('score', 0),
            total_questions=data.get('total_questions', 0),
            answers=data.get('answers', {}),
            generated_questions=data.get('generated_questions', [])
        )
        
        test_result.save()
        
        return jsonify({'message': 'Test result saved successfully', 'id': str(test_result.id)}), 201
    except Exception as e:
        return jsonify({'message': f'Error saving test result: {str(e)}'}), 500


@app.route('/api/user-test-results', methods=['GET'])
@login_required
def get_user_test_results():
    """Get all test results for current user"""
    try:
        results = TestResult.objects(user=current_user).order_by('-created_at')
        
        return jsonify({
            'results': [{
                'id': str(r.id),
                'topic': r.topic,
                'score': r.score,
                'total_questions': r.total_questions,
                'created_at': r.created_at.isoformat()
            } for r in results]
        }), 200
    except Exception as e:
        return jsonify({'message': f'Error retrieving results: {str(e)}'}), 500


if __name__ == '__main__':

    print("  Starting Medical Question Generator API...")
    print("  ML Models Status:")
    print(f"   Voting Ensemble: {' Loaded' if question_api.voting_model else ' Not loaded'}")
    print(f"   Preprocessing: {' Loaded' if question_api.preprocessing else ' Not loaded'}")
    print("\n Server starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
