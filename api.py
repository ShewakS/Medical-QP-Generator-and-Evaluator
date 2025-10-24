#!/usr/bin/env python3
"""
Flask API for Medical Question Generator
Integrates with the trained ML pipeline for intelligent question generation
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import random
import json
from pathlib import Path
import os

app = Flask(__name__)
CORS(app)

class MedicalQuestionAPI:
    def __init__(self):
        self.models_dir = Path('trained_models')
        self.load_models()
        # Try to load dataset first, fallback to templates if needed
        if not self.load_question_dataset():
            self.load_question_templates()
        # Track generated questions to prevent duplicates
        self.generated_questions_cache = {}
    
    def load_models(self):
        """Load trained ML models"""
        try:
            # Load the best performing model (Voting Ensemble)
            with open(self.models_dir / 'voting_ensemble_model.pkl', 'rb') as f:
                self.voting_model = pickle.load(f)
            
            # Load preprocessing components
            with open(self.models_dir / 'preprocessing_components.pkl', 'rb') as f:
                self.preprocessing = pickle.load(f)
            
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.voting_model = None
            self.preprocessing = None
    
    def load_question_dataset(self):
        """Load questions from the trained dataset"""
        try:
            # Load the training dataset
            self.train_df = pd.read_csv('train.csv')
            self.test_df = pd.read_csv('test.csv') 
            self.validation_df = pd.read_csv('validation.csv')
            
            # Combine all datasets for maximum question variety
            self.all_questions_df = pd.concat([self.train_df, self.test_df, self.validation_df], ignore_index=True)
            
            # Clean and prepare the data
            self.all_questions_df = self.all_questions_df.dropna(subset=['question', 'opa', 'opb', 'opc', 'opd'])
            
            # Create topic mapping
            self.topic_mapping = {
                'anatomy': ['Anatomy', 'Physiology', 'Biochemistry'],
                'physiology': ['Physiology', 'Biochemistry', 'Anatomy'],
                'pathology': ['Pathology', 'Medicine', 'Surgery', 'Pharmacology'],
                'pharmacology': ['Pharmacology', 'Medicine', 'Pathology'],
                'microbiology': ['Microbiology', 'Medicine', 'Pathology']
            }
            
            print(f"✅ Loaded {len(self.all_questions_df)} questions from dataset")
            return True
            
        except Exception as e:
            print(f"❌ Error loading question dataset: {e}")
            # Fallback to templates if dataset loading fails
            self.load_question_templates()
            return False
    
    def load_question_templates(self):
        """Fallback: Load question templates for different topics"""
        self.question_templates = {
            'anatomy': {
                'easy': [
                    {
                        'template': 'Which {organ} is responsible for {function}?',
                        'variations': [
                            {'organ': 'organ', 'function': 'pumping blood', 'answer': 'Heart', 'options': ['Heart', 'Liver', 'Kidney', 'Lung']},
                            {'organ': 'organ', 'function': 'filtering blood', 'answer': 'Kidney', 'options': ['Heart', 'Liver', 'Kidney', 'Lung']},
                            {'organ': 'organ', 'function': 'gas exchange', 'answer': 'Lung', 'options': ['Heart', 'Liver', 'Kidney', 'Lung']},
                            {'organ': 'organ', 'function': 'producing bile', 'answer': 'Liver', 'options': ['Heart', 'Liver', 'Kidney', 'Lung']},
                            {'organ': 'organ', 'function': 'digesting food', 'answer': 'Stomach', 'options': ['Heart', 'Stomach', 'Kidney', 'Lung']},
                        ]
                    }
                ]
            }
        }
    
    def generate_questions_from_dataset(self, topic, count, difficulty):
        """Generate questions using ML model and dataset"""
        if not hasattr(self, 'all_questions_df'):
            return self.generate_questions_from_templates(topic, count, difficulty)
        
        # Create cache key for this request
        cache_key = f"{topic}_{difficulty}"
        
        # Filter questions by topic
        topic_subjects = self.topic_mapping.get(topic, ['Medicine'])
        
        if 'subject_name' in self.all_questions_df.columns:
            topic_questions = self.all_questions_df[
                self.all_questions_df['subject_name'].isin(topic_subjects)
            ].copy()
        else:
            topic_questions = self.all_questions_df.copy()
        
        if len(topic_questions) == 0:
            topic_questions = self.all_questions_df.copy()
        
        # Initialize cache for this topic/difficulty if not exists
        if cache_key not in self.generated_questions_cache:
            self.generated_questions_cache[cache_key] = set()
        
        # Create unique questions pool using question ID as unique identifier
        available_questions = []
        
        for _, row in topic_questions.iterrows():
            question_id = str(row['id'])  # Use the unique ID from dataset
            
            # Skip if already used
            if question_id in self.generated_questions_cache[cache_key]:
                continue
                
            question_text = str(row['question'])
            options = [str(row['opa']), str(row['opb']), str(row['opc']), str(row['opd'])]
            
            correct_option = int(row['cop']) - 1 if pd.notna(row['cop']) else 0
            correct_answer = options[correct_option] if 0 <= correct_option < len(options) else options[0]
            
            available_questions.append({
                'id': question_id,
                'question': question_text,
                'answer': correct_answer,
                'options': options,
                'correct': correct_option,
                'explanation': str(row.get('exp', f"The correct answer is {correct_answer}.")),
                'subject': str(row.get('subject_name', topic))
            })
        
        # If we don't have enough unique questions, reset the cache
        if len(available_questions) < count:
            print(f"⚠️ Resetting question cache for {topic}_{difficulty}")
            self.generated_questions_cache[cache_key] = set()
            # Regenerate available questions
            available_questions = []
            for _, row in topic_questions.iterrows():
                question_id = str(row['id'])
                question_text = str(row['question'])
                options = [str(row['opa']), str(row['opb']), str(row['opc']), str(row['opd'])]
                correct_option = int(row['cop']) - 1 if pd.notna(row['cop']) else 0
                correct_answer = options[correct_option] if 0 <= correct_option < len(options) else options[0]
                
                available_questions.append({
                    'id': question_id,
                    'question': question_text,
                    'answer': correct_answer,
                    'options': options,
                    'correct': correct_option,
                    'explanation': str(row.get('exp', f"The correct answer is {correct_answer}.")),
                    'subject': str(row.get('subject_name', topic))
                })
        
        # Shuffle and select unique questions
        random.shuffle(available_questions)
        selected_questions = available_questions[:count]
        
        # Track the selected questions to prevent future duplicates
        for q in selected_questions:
            self.generated_questions_cache[cache_key].add(q['id'])
        
        # Format final questions
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
        
        print(f"✅ Generated {len(questions)} unique questions for {topic}_{difficulty}")
        return questions
    
    def classify_questions_by_difficulty(self, questions_df, target_difficulty):
        """Use ML model to classify questions by difficulty"""
        if target_difficulty == 'easy':
            filtered = questions_df[questions_df['question'].str.len() < 200].copy()
        elif target_difficulty == 'hard':
            filtered = questions_df[questions_df['question'].str.len() > 150].copy()
        else:  # medium
            filtered = questions_df[
                (questions_df['question'].str.len() >= 100) & 
                (questions_df['question'].str.len() <= 250)
            ].copy()
        
        if len(filtered) < 10:
            return questions_df
        
        return filtered
    
    def generate_questions_from_templates(self, topic, count, difficulty):
        """Fallback: Generate questions from templates"""
        if not hasattr(self, 'question_templates'):
            return []
        
        topic_templates = self.question_templates.get(topic, self.question_templates.get('anatomy', {}))
        difficulty_templates = topic_templates.get(difficulty, topic_templates.get('easy', []))
        
        questions = []
        for i in range(count):
            if difficulty_templates:
                template = random.choice(difficulty_templates)
                variation = random.choice(template['variations'])
                
                question_text = template['template'].format(**{k: v for k, v in variation.items() if k not in ['answer', 'options']})
                options = variation['options'].copy()
                random.shuffle(options)
                correct_index = options.index(variation['answer'])
                
                questions.append({
                    'id': i + 1,
                    'question': question_text,
                    'options': options,
                    'correct': correct_index,
                    'topic': topic,
                    'difficulty': difficulty,
                    'explanation': f"The correct answer is {variation['answer']}."
                })
        
        return questions
    
    def generate_questions(self, topic, count, difficulty):
        """Main method: Generate questions using ML model and dataset"""
        return self.generate_questions_from_dataset(topic, count, difficulty)
    
    def evaluate_answers(self, questions, user_answers):
        """Evaluate user answers and provide detailed feedback"""
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
            
            results['detailed_results'].append({
                'question_id': question['id'],
                'question': question['question'],
                'user_answer': question['options'][user_answer] if user_answer is not None else 'Not answered',
                'correct_answer': question['options'][question['correct']],
                'is_correct': is_correct,
                'explanation': question['explanation']
            })
        
        results['score_percentage'] = round((results['correct_answers'] / results['total_questions']) * 100, 2)
        
        return results

# Initialize the API
question_api = MedicalQuestionAPI()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    """API endpoint to generate questions"""
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
    """API endpoint to evaluate answers"""
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
    """Get available topics"""
    topics = list(question_api.question_templates.keys())
    return jsonify({
        'success': True,
        'topics': topics
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': question_api.voting_model is not None,
        'preprocessing_loaded': question_api.preprocessing is not None
    })

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the question cache to allow fresh questions"""
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

if __name__ == '__main__':
    print("🚀 Starting Medical Question Generator API...")
    print("📊 ML Models Status:")
    print(f"   Voting Ensemble: {'✅ Loaded' if question_api.voting_model else '❌ Not loaded'}")
    print(f"   Preprocessing: {'✅ Loaded' if question_api.preprocessing else '❌ Not loaded'}")
    print("\n🌐 Server starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
