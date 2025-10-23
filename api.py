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
        self.load_question_templates()
    
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
    
    def load_question_templates(self):
        """Load question templates for different topics"""
        self.question_templates = {
            'anatomy': {
                'easy': [
                    {
                        'template': 'Which {organ} is responsible for {function}?',
                        'variations': [
                            {'organ': 'organ', 'function': 'pumping blood', 'answer': 'Heart', 'options': ['Heart', 'Liver', 'Kidney', 'Lung']},
                            {'organ': 'organ', 'function': 'filtering blood', 'answer': 'Kidney', 'options': ['Heart', 'Liver', 'Kidney', 'Lung']},
                            {'organ': 'organ', 'function': 'gas exchange', 'answer': 'Lung', 'options': ['Heart', 'Liver', 'Kidney', 'Lung']},
                        ]
                    },
                    {
                        'template': 'How many {structure} does the human body have?',
                        'variations': [
                            {'structure': 'chambers in the heart', 'answer': '4', 'options': ['2', '3', '4', '5']},
                            {'structure': 'lobes in the liver', 'answer': '4', 'options': ['2', '3', '4', '5']},
                            {'structure': 'kidneys', 'answer': '2', 'options': ['1', '2', '3', '4']},
                        ]
                    }
                ],
                'medium': [
                    {
                        'template': 'Which bone is the {characteristic} in the human body?',
                        'variations': [
                            {'characteristic': 'longest', 'answer': 'Femur', 'options': ['Femur', 'Tibia', 'Humerus', 'Radius']},
                            {'characteristic': 'smallest', 'answer': 'Stapes', 'options': ['Stapes', 'Malleus', 'Incus', 'Cochlea']},
                        ]
                    }
                ],
                'hard': [
                    {
                        'template': 'Which nerve controls {function}?',
                        'variations': [
                            {'function': 'facial expression', 'answer': 'Facial nerve (VII)', 'options': ['Facial nerve (VII)', 'Trigeminal nerve (V)', 'Oculomotor nerve (III)', 'Vagus nerve (X)']},
                        ]
                    }
                ]
            },
            'physiology': {
                'easy': [
                    {
                        'template': 'What is the normal {parameter} for adults?',
                        'variations': [
                            {'parameter': 'heart rate', 'answer': '60-100 bpm', 'options': ['40-60 bpm', '60-100 bpm', '100-120 bpm', '120-140 bpm']},
                            {'parameter': 'body temperature', 'answer': '98.6°F (37°C)', 'options': ['96.8°F (36°C)', '98.6°F (37°C)', '100.4°F (38°C)', '102.2°F (39°C)']},
                        ]
                    }
                ],
                'medium': [
                    {
                        'template': 'Which hormone regulates {function}?',
                        'variations': [
                            {'function': 'blood sugar', 'answer': 'Insulin', 'options': ['Insulin', 'Cortisol', 'Thyroxine', 'Adrenaline']},
                            {'function': 'metabolism', 'answer': 'Thyroxine', 'options': ['Insulin', 'Cortisol', 'Thyroxine', 'Adrenaline']},
                        ]
                    }
                ],
                'hard': [
                    {
                        'template': 'What is the mechanism of {process}?',
                        'variations': [
                            {'process': 'muscle contraction', 'answer': 'Sliding filament theory', 'options': ['Sliding filament theory', 'Cross-bridge cycling', 'Calcium release', 'ATP hydrolysis']},
                        ]
                    }
                ]
            },
            'pathology': {
                'easy': [
                    {
                        'template': 'What condition is characterized by {symptom}?',
                        'variations': [
                            {'symptom': 'high blood pressure', 'answer': 'Hypertension', 'options': ['Hypertension', 'Hypotension', 'Tachycardia', 'Bradycardia']},
                            {'symptom': 'low blood sugar', 'answer': 'Hypoglycemia', 'options': ['Hyperglycemia', 'Hypoglycemia', 'Diabetes', 'Insulin resistance']},
                        ]
                    }
                ],
                'medium': [
                    {
                        'template': 'Which is the most common type of {disease_category}?',
                        'variations': [
                            {'disease_category': 'cancer', 'answer': 'Lung cancer', 'options': ['Lung cancer', 'Breast cancer', 'Prostate cancer', 'Colorectal cancer']},
                            {'disease_category': 'heart disease', 'answer': 'Coronary artery disease', 'options': ['Coronary artery disease', 'Heart failure', 'Arrhythmia', 'Valve disease']},
                        ]
                    }
                ],
                'hard': [
                    {
                        'template': 'What is the pathophysiology of {condition}?',
                        'variations': [
                            {'condition': 'atherosclerosis', 'answer': 'Plaque buildup in arteries', 'options': ['Plaque buildup in arteries', 'Blood clot formation', 'Vessel inflammation', 'Genetic mutation']},
                        ]
                    }
                ]
            },
            'pharmacology': {
                'easy': [
                    {
                        'template': 'What is the generic name for {brand_name}?',
                        'variations': [
                            {'brand_name': 'Tylenol', 'answer': 'Acetaminophen', 'options': ['Acetaminophen', 'Ibuprofen', 'Aspirin', 'Naproxen']},
                            {'brand_name': 'Advil', 'answer': 'Ibuprofen', 'options': ['Acetaminophen', 'Ibuprofen', 'Aspirin', 'Naproxen']},
                        ]
                    }
                ],
                'medium': [
                    {
                        'template': 'Which class of drugs is used to treat {condition}?',
                        'variations': [
                            {'condition': 'bacterial infections', 'answer': 'Antibiotics', 'options': ['Antibiotics', 'Antivirals', 'Antifungals', 'Antihistamines']},
                            {'condition': 'high blood pressure', 'answer': 'Antihypertensives', 'options': ['Antihypertensives', 'Diuretics', 'Beta-blockers', 'ACE inhibitors']},
                        ]
                    }
                ],
                'hard': [
                    {
                        'template': 'What is the mechanism of action of {drug_class}?',
                        'variations': [
                            {'drug_class': 'ACE inhibitors', 'answer': 'Block angiotensin-converting enzyme', 'options': ['Block angiotensin-converting enzyme', 'Block calcium channels', 'Block beta receptors', 'Block sodium channels']},
                        ]
                    }
                ]
            }
        }
    
    def generate_questions(self, topic, count, difficulty):
        """Generate questions based on topic, count, and difficulty"""
        questions = []
        
        # Get templates for the topic and difficulty
        topic_templates = self.question_templates.get(topic, self.question_templates['anatomy'])
        difficulty_templates = topic_templates.get(difficulty, topic_templates['easy'])
        
        # Generate questions
        for i in range(count):
            template = random.choice(difficulty_templates)
            variation = random.choice(template['variations'])
            
            # Create question
            question_text = template['template'].format(**{k: v for k, v in variation.items() if k not in ['answer', 'options']})
            
            # Shuffle options
            options = variation['options'].copy()
            random.shuffle(options)
            
            # Find correct answer index
            correct_index = options.index(variation['answer'])
            
            question = {
                'id': i + 1,
                'question': question_text,
                'options': options,
                'correct': correct_index,
                'topic': topic,
                'difficulty': difficulty,
                'explanation': f"The correct answer is {variation['answer']}."
            }
            
            questions.append(question)
        
        return questions
    
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
        
        # Validate inputs
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        if count < 1 or count > 20:
            return jsonify({'error': 'Question count must be between 1 and 20'}), 400
        
        # Generate questions
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
        
        # Evaluate answers
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

if __name__ == '__main__':
    print("🚀 Starting Medical Question Generator API...")
    print("📊 ML Models Status:")
    print(f"   Voting Ensemble: {'✅ Loaded' if question_api.voting_model else '❌ Not loaded'}")
    print(f"   Preprocessing: {'✅ Loaded' if question_api.preprocessing else '❌ Not loaded'}")
    print("\n🌐 Server starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
