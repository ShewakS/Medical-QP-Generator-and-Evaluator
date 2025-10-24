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
                    },
                    {
                        'template': 'How many {structure} does the human body have?',
                        'variations': [
                            {'structure': 'chambers in the heart', 'answer': '4', 'options': ['2', '3', '4', '5']},
                            {'structure': 'lobes in the liver', 'answer': '4', 'options': ['2', '3', '4', '5']},
                            {'structure': 'kidneys', 'answer': '2', 'options': ['1', '2', '3', '4']},
                            {'structure': 'lungs', 'answer': '2', 'options': ['1', '2', '3', '4']},
                            {'structure': 'eyes', 'answer': '2', 'options': ['1', '2', '3', '4']},
                        ]
                    },
                    {
                        'template': 'What is the main function of the {organ}?',
                        'variations': [
                            {'organ': 'brain', 'answer': 'Control body functions', 'options': ['Control body functions', 'Pump blood', 'Filter waste', 'Produce hormones']},
                            {'organ': 'spine', 'answer': 'Support the body', 'options': ['Support the body', 'Pump blood', 'Filter waste', 'Digest food']},
                            {'organ': 'ribs', 'answer': 'Protect internal organs', 'options': ['Protect internal organs', 'Pump blood', 'Filter waste', 'Produce energy']},
                        ]
                    },
                    {
                        'template': 'Where is the {organ} located in the human body?',
                        'variations': [
                            {'organ': 'heart', 'answer': 'Chest cavity', 'options': ['Chest cavity', 'Abdominal cavity', 'Head', 'Back']},
                            {'organ': 'brain', 'answer': 'Head', 'options': ['Chest cavity', 'Abdominal cavity', 'Head', 'Back']},
                            {'organ': 'stomach', 'answer': 'Abdominal cavity', 'options': ['Chest cavity', 'Abdominal cavity', 'Head', 'Back']},
                        ]
                    }
                ],
                'medium': [
                    {
                        'template': 'Which bone is the {characteristic} in the human body?',
                        'variations': [
                            {'characteristic': 'strongest', 'answer': 'Femur', 'options': ['Femur', 'Tibia', 'Humerus', 'Radius']},
                            {'characteristic': 'smallest', 'answer': 'Stapes', 'options': ['Stapes', 'Malleus', 'Incus', 'Cochlea']},
                        ]
                    },
                    {
                        'template': 'Which part of the {system} is responsible for {function}?',
                        'variations': [
                            {'system': 'skeletal system', 'function': 'protecting the brain', 'answer': 'Skull', 'options': ['Skull', 'Spine', 'Ribs', 'Pelvis']},
                            {'system': 'muscular system', 'function': 'facial expressions', 'answer': 'Facial muscles', 'options': ['Facial muscles', 'Neck muscles', 'Jaw muscles', 'Eye muscles']},
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
                            {'parameter': 'blood pressure', 'answer': '120/80 mmHg', 'options': ['90/60 mmHg', '120/80 mmHg', '140/90 mmHg', '160/100 mmHg']},
                        ]
                    },
                    {
                        'template': 'Which system controls {function}?',
                        'variations': [
                            {'function': 'breathing', 'answer': 'Respiratory system', 'options': ['Respiratory system', 'Circulatory system', 'Nervous system', 'Digestive system']},
                            {'function': 'digestion', 'answer': 'Digestive system', 'options': ['Respiratory system', 'Circulatory system', 'Nervous system', 'Digestive system']},
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
    
    def generate_questions_from_dataset(self, topic, count, difficulty):
        """Generate questions using ML model and dataset"""
        if not hasattr(self, 'all_questions_df'):
            # Fallback to template method if dataset not loaded
            return self.generate_questions_from_templates(topic, count, difficulty)
        
        # Create cache key for this request
        cache_key = f"{topic}_{difficulty}"
        
        # Filter questions by topic
        topic_subjects = self.topic_mapping.get(topic, ['Medicine'])
        
        # Filter dataset by subject
        if 'subject_name' in self.all_questions_df.columns:
            topic_questions = self.all_questions_df[
                self.all_questions_df['subject_name'].isin(topic_subjects)
            ].copy()
        else:
            # If no subject column, use all questions
            topic_questions = self.all_questions_df.copy()
        
        if len(topic_questions) == 0:
            print(f"⚠️ No questions found for topic {topic}, using all questions")
            topic_questions = self.all_questions_df.copy()
        
        # Use ML model to classify questions by difficulty if available
        if self.voting_model is not None and self.preprocessing is not None:
            try:
                # Prepare questions for ML classification
                topic_questions = self.classify_questions_by_difficulty(topic_questions, difficulty)
            except Exception as e:
                print(f"⚠️ ML classification failed: {e}, using random selection")
        
        # Create unique questions pool
        all_possible_questions = []
        question_hashes = set()
        
        for _, row in topic_questions.iterrows():
            question_text = str(row['question'])
            options = [str(row['opa']), str(row['opb']), str(row['opc']), str(row['opd'])]
            
            # Get correct answer
            correct_option = int(row['cop']) - 1 if pd.notna(row['cop']) else 0
            correct_answer = options[correct_option] if 0 <= correct_option < len(options) else options[0]
            
            # Create multiple variations by shuffling options
            for shuffle_seed in range(3):  # Create 3 variations per question
                shuffled_options = options.copy()
                temp_random = random.Random(shuffle_seed)
                temp_random.shuffle(shuffled_options)
                
                # Find new correct index
                try:
                    new_correct_index = shuffled_options.index(correct_answer)
                except ValueError:
                    new_correct_index = 0
                
                # Create unique identifier
                question_hash = hash(f"{question_text}_{'-'.join(shuffled_options)}")
                
                if question_hash not in question_hashes:
                    question_hashes.add(question_hash)
                    
                    all_possible_questions.append({
                        'question': question_text,
                        'answer': correct_answer,
                        'options': shuffled_options,
                        'correct': new_correct_index,
                        'hash': question_hash,
                        'explanation': str(row.get('exp', f"The correct answer is {correct_answer}.")),
                        'subject': str(row.get('subject_name', topic))
                    })
        
        # Initialize cache for this topic/difficulty if not exists
        if cache_key not in self.generated_questions_cache:
            self.generated_questions_cache[cache_key] = set()
        
        # Filter out previously generated questions
        available_questions = [
            q for q in all_possible_questions 
            if q['hash'] not in self.generated_questions_cache[cache_key]
        ]
        
        # If we don't have enough unique questions, reset the cache
        if len(available_questions) < count:
            print(f"⚠️ Resetting question cache for {topic}_{difficulty} - generating fresh questions")
            self.generated_questions_cache[cache_key] = set()
            available_questions = all_possible_questions.copy()
        
        # Shuffle and select unique questions
        random.shuffle(available_questions)
        selected_questions = available_questions[:count]
        
        # Track the selected questions to prevent future duplicates
        for q in selected_questions:
            self.generated_questions_cache[cache_key].add(q['hash'])
        
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
        
        print(f"✅ Generated {len(questions)} ML-based questions for {topic}_{difficulty}")
        return questions
    
    def classify_questions_by_difficulty(self, questions_df, target_difficulty):
        """Use ML model to classify questions by difficulty"""
        # This is a simplified difficulty classification
        # You can enhance this based on your specific ML model capabilities
        
        if target_difficulty == 'easy':
            # Select shorter, simpler questions for easy
            filtered = questions_df[questions_df['question'].str.len() < 200].copy()
        elif target_difficulty == 'hard':
            # Select longer, more complex questions for hard
            filtered = questions_df[questions_df['question'].str.len() > 150].copy()
        else:  # medium
            # Select medium-length questions
            filtered = questions_df[
                (questions_df['question'].str.len() >= 100) & 
                (questions_df['question'].str.len() <= 250)
            ].copy()
        
        # If filtered set is too small, return original
        if len(filtered) < 10:
            return questions_df
        
        return filtered
    
    def generate_questions_from_templates(self, topic, count, difficulty):
        """Fallback: Generate questions from templates"""
        if not hasattr(self, 'question_templates'):
            return []
        
        # Original template-based logic (simplified)
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
            # Clear all cache
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
