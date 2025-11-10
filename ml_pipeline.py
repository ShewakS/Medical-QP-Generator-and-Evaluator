import pandas as pd
import numpy as np
import time
import psutil
import gc
import pickle
import re
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class OptimizedMLPipeline:
    def __init__(self, data_dir: str = ".", n_jobs: int = -1, verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.n_jobs = n_jobs if n_jobs != -1 else psutil.cpu_count()
        self.verbose = verbose
        self.vectorizer = None
        self.pca = None
        self.label_encoder = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        if self.verbose:
            print(f" Pipeline initialized with {self.n_jobs} CPU cores")
    
    def _download_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            if self.verbose:
                print(" Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def _log_memory_usage(self, step: str):
        if self.verbose:
            memory = psutil.virtual_memory()
            print(f" {step}: {memory.percent:.1f}% memory used ({memory.used/1024**3:.1f}GB)")
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 1-2: Loading Datasets")
            print("="*50)
        start_time = time.time()
        dtype_dict = {
            'id': 'string', 'question': 'string', 'opa': 'string', 'opb': 'string', 
            'opc': 'string', 'opd': 'string', 'cop': 'int8', 'choice_type': 'category',
            'exp': 'string', 'subject_name': 'category', 'topic_name': 'category'
        }
        train_df = pd.read_csv(self.data_dir / 'train.csv', dtype=dtype_dict)
        test_df = pd.read_csv(self.data_dir / 'test.csv', dtype=dtype_dict)
        val_df = pd.read_csv(self.data_dir / 'validation.csv', dtype=dtype_dict)
        load_time = time.time() - start_time
        if self.verbose:
            print(f" Loaded datasets in {load_time:.2f}s")
            print(f" Train: {len(train_df):,} samples")
            print(f" Test: {len(test_df):,} samples") 
            print(f" Validation: {len(val_df):,} samples")
            print(f" Total: {len(train_df) + len(test_df) + len(val_df):,} samples")
        self._log_memory_usage("After loading")
        return train_df, test_df, val_df
    
    def inspect_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame):
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 3: Data Inspection")
            print("="*50)
        datasets = {'Train': train_df, 'Test': test_df, 'Validation': val_df}
        for name, df in datasets.items():
            print(f"\n {name} Dataset:")
            print(f"   Shape: {df.shape}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"     Missing values: {missing[missing > 0].to_dict()}")
            else:
                print("    No missing values")
            duplicates = df.duplicated().sum()
            print(f"    Duplicates: {duplicates}")
            if 'subject_name' in df.columns:
                print(f"    Subjects: {df['subject_name'].nunique()}")
            if 'choice_type' in df.columns:
                print(f"    Choice types: {df['choice_type'].value_counts().to_dict()}")
    
    def _clean_text_fast(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def clean_and_preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 4: Data Cleaning & Preprocessing")
            print("="*50)
        start_time = time.time()
        def process_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
            df = df.copy()
            text_columns = ['question', 'opa', 'opb', 'opc', 'opd', 'exp']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('')
            if 'subject_name' in df.columns:
                if df['subject_name'].dtype.name == 'category':
                    if 'Unknown' not in df['subject_name'].cat.categories:
                        df['subject_name'] = df['subject_name'].cat.add_categories(['Unknown'])
                    df['subject_name'] = df['subject_name'].fillna('Unknown')
                else:
                    df['subject_name'] = df['subject_name'].fillna('Unknown')
            if 'topic_name' in df.columns:
                if df['topic_name'].dtype.name == 'category':
                    if 'Unknown' not in df['topic_name'].cat.categories:
                        df['topic_name'] = df['topic_name'].cat.add_categories(['Unknown'])
                    df['topic_name'] = df['topic_name'].fillna('Unknown')
                else:
                    df['topic_name'] = df['topic_name'].fillna('Unknown')
            initial_len = len(df)
            df = df.drop_duplicates()
            removed = initial_len - len(df)
            if self.verbose:
                print(f"    Cleaning {name} dataset...")
            for col in text_columns:
                if col in df.columns:
                    if self.verbose:
                        print(f"     Cleaning {col}...")
                    df[col] = df[col].apply(self._clean_text_fast)
            if self.verbose and removed > 0:
                print(f"     Removed {removed} duplicates from {name}")
            return df
        train_clean = process_dataset(train_df, "Train")
        test_clean = process_dataset(test_df, "Test") 
        val_clean = process_dataset(val_df, "Validation")
        gc.collect()
        clean_time = time.time() - start_time
        if self.verbose:
            print(f" Cleaning completed in {clean_time:.2f}s")
        self._log_memory_usage("After cleaning")
        return train_clean, test_clean, val_clean
    
    def encode_categorical_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                  val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.verbose:
            print("\n" + "="*50)
            print("  STEP 5: Categorical Feature Encoding")
            print("="*50)
        start_time = time.time()
        all_data = pd.concat([train_df, test_df, val_df], ignore_index=True)
        categorical_cols = ['subject_name', 'choice_type']
        encoders = {}
        for col in categorical_cols:
            if col in all_data.columns:
                encoder = LabelEncoder()
                all_data[f'{col}_encoded'] = encoder.fit_transform(all_data[col].astype(str))
                encoders[col] = encoder
                if self.verbose:
                    print(f"    Encoded {col}: {len(encoder.classes_)} unique values")
        train_len = len(train_df)
        test_len = len(test_df)
        train_encoded = all_data[:train_len].copy()
        test_encoded = all_data[train_len:train_len+test_len].copy()
        val_encoded = all_data[train_len+test_len:].copy()
        self.categorical_encoders = encoders
        encode_time = time.time() - start_time
        if self.verbose:
            print(f" Encoding completed in {encode_time:.2f}s")
        return train_encoded, test_encoded, val_encoded
    
    def apply_tfidf_vectorization(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 6: TF-IDF Vectorization")
            print("="*50)
        start_time = time.time()
        def combine_text(row):
            question = str(row.get('question', ''))
            opa = str(row.get('opa', ''))
            opb = str(row.get('opb', ''))
            opc = str(row.get('opc', ''))
            opd = str(row.get('opd', ''))
            text_parts = [question, opa, opb, opc, opd]
            combined_text = ' '.join([part for part in text_parts if part])
            question_lower = question.lower()
            all_options = ' '.join([opa, opb, opc, opd]).lower()
            ultra_multi = ['all of the above', 'all the above', 'none of the above', 'except', 'not true', 
                          'false statement', 'incorrect statement', 'not include', 'not associated',
                          'not characteristic', 'not seen in', 'not found in', 'all are true', 'all are correct',
                          'all following are true', 'following are all true', 'all statements are',
                          'most appropriate', 'best describes', 'most likely', 'least likely', 'commonest',
                          'most common', 'rarest', 'most rare', 'best treatment', 'first line', 'drug of choice']
            ultra_single = ['what is the', 'which is the', 'define', 'definition of', 'meaning of', 
                           'caused by', 'due to', 'result of', 'characterized by', 'typical of',
                           'classic sign', 'pathognomonic', 'diagnostic of', 'specific for',
                           'gold standard', 'investigation of choice', 'treatment of choice']
            ultra_multi_count = sum(1 for indicator in ultra_multi if indicator in question_lower)
            ultra_single_count = sum(1 for indicator in ultra_single if indicator in question_lower)
            if ultra_multi_count > 0:
                combined_text += ' ULTRA_MULTI_CHOICE ' * (ultra_multi_count * 10)
            if ultra_single_count > 0:
                combined_text += ' ULTRA_SINGLE_CHOICE ' * (ultra_single_count * 10)
            strong_multi = ['all', 'most', 'least', 'best', 'worst', 'always', 'never', 'only',
                           'true statement', 'correct statement', 'false about', 'incorrect about',
                           'not seen', 'not true about', 'contraindicated', 'side effect']
            strong_single = ['which', 'what', 'who', 'when', 'where', 'how', 'why', 'name the',
                           'identify', 'recognize', 'diagnose', 'classify', 'type of', 'kind of']
            strong_multi_count = sum(1 for word in strong_multi if word in question_lower)
            strong_single_count = sum(1 for word in strong_single if word in question_lower)
            if strong_multi_count > 0:
                combined_text += ' STRONG_MULTI_CHOICE ' * (strong_multi_count * 5)
            if strong_single_count > 0:
                combined_text += ' STRONG_SINGLE_CHOICE ' * (strong_single_count * 5)
            if 'all of the above' in all_options or 'none of the above' in all_options:
                combined_text += ' OPTIONS_MULTI_PATTERN ' * 8
            if question.count(',') > 2:
                combined_text += ' COMPLEX_STRUCTURE ' * 3
            if question.count('(') > 0 and question.count(')') > 0:
                combined_text += ' HAS_PARENTHESES ' * 2
            if any(char.isdigit() for char in question):
                combined_text += ' HAS_NUMBERS ' * 2
            medical_multi_terms = ['syndrome', 'disease', 'condition', 'disorder', 'complications',
                                 'manifestations', 'symptoms', 'signs', 'features', 'characteristics']
            medical_single_terms = ['organism', 'bacteria', 'virus', 'drug', 'medication', 'treatment',
                                  'procedure', 'investigation', 'test', 'diagnosis']
            if any(term in question_lower for term in medical_multi_terms):
                combined_text += ' MEDICAL_MULTI_TERM ' * 3
            if any(term in question_lower for term in medical_single_terms):
                combined_text += ' MEDICAL_SINGLE_TERM ' * 3
            word_count = len(question.split())
            if word_count > 25:
                combined_text += ' VERY_LONG_QUESTION ' * 4
            elif word_count > 15:
                combined_text += ' LONG_QUESTION ' * 2
            elif word_count < 8:
                combined_text += ' SHORT_QUESTION ' * 2
            if question.count('?') > 1:
                combined_text += ' MULTIPLE_QUESTIONS ' * 3
            if question.count(':') > 0:
                combined_text += ' HAS_COLON ' * 2
            if question.count(';') > 0:
                combined_text += ' HAS_SEMICOLON ' * 2
            return combined_text
        train_text = train_df.apply(combine_text, axis=1)
        test_text = test_df.apply(combine_text, axis=1)
        val_text = val_df.apply(combine_text, axis=1)
        self.vectorizer = TfidfVectorizer(
            max_features=25000, min_df=1, max_df=0.95, stop_words=None, ngram_range=(1, 5),
            lowercase=True, strip_accents='ascii', token_pattern=r'\b[a-zA-Z_]{2,}\b',
            sublinear_tf=True, use_idf=True, smooth_idf=True, norm='l2', analyzer='word', binary=False
        )
        if self.verbose:
            print("    Fitting TF-IDF vectorizer...")
        X_train_tfidf = self.vectorizer.fit_transform(train_text)
        X_test_tfidf = self.vectorizer.transform(test_text)
        X_val_tfidf = self.vectorizer.transform(val_text)
        tfidf_time = time.time() - start_time
        if self.verbose:
            print(f" TF-IDF completed in {tfidf_time:.2f}s")
            print(f" Feature matrix shape: {X_train_tfidf.shape}")
            print(f" Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            print(f" Sparsity: {(1 - X_train_tfidf.nnz / np.prod(X_train_tfidf.shape)) * 100:.1f}%")
        self._log_memory_usage("After TF-IDF")
        return X_train_tfidf, X_test_tfidf, X_val_tfidf
    
    def apply_feature_selection(self, X_train: np.ndarray, y_train: np.ndarray, 
                               X_val: np.ndarray, X_test: np.ndarray, 
                               k_features: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.verbose:
            print("\n" + "="*50)
            print(" FEATURE SELECTION")
            print("="*50)
        start_time = time.time()
        if self.verbose:
            print(f"    Selecting top {k_features} features using mutual information...")
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)
        selection_time = time.time() - start_time
        if self.verbose:
            print(f" Feature selection completed in {selection_time:.2f}s")
            print(f" Original features: {X_train.shape[1]}")
            print(f" Selected features: {X_train_selected.shape[1]}")
            print(f" Reduction: {(1 - X_train_selected.shape[1]/X_train.shape[1]) * 100:.1f}%")
        return X_train_selected, X_val_selected, X_test_selected
    
    def split_data(self, X_train: np.ndarray, train_df: pd.DataFrame, 
                  test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.verbose:
            print("\n" + "="*50)
            print("  STEP 7: Data Splitting")
            print("="*50)
        y = train_df['choice_type_encoded'] if 'choice_type_encoded' in train_df.columns else train_df['choice_type']
        if not hasattr(self, 'label_encoder') or self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y.astype(str))
        else:
            y = y.values if hasattr(y, 'values') else y
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y, test_size=test_size, random_state=42, stratify=y
        )
        if self.verbose:
            print(f" Data split completed")
            print(f"    Training set: {X_train_split.shape[0]:,} samples")
            print(f"    Validation set: {X_val_split.shape[0]:,} samples")
            print(f"    Number of classes: {len(np.unique(y))}")
        return X_train_split, X_val_split, y_train_split, y_val_split
    
    def apply_pca(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, 
                 n_components: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 8: PCA Dimensionality Reduction")
            print("="*50)
        start_time = time.time()
        if hasattr(X_train, 'toarray'):
            X_train_dense = X_train.toarray()
            X_val_dense = X_val.toarray()
            X_test_dense = X_test.toarray()
        else:
            X_train_dense = X_train
            X_val_dense = X_val
            X_test_dense = X_test
        if self.verbose:
            print("    Applying feature scaling...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_dense)
        X_val_scaled = scaler.transform(X_val_dense)
        X_test_scaled = scaler.transform(X_test_dense)
        self.feature_scaler = scaler
        self.pca = PCA(n_components=n_components, random_state=42)
        if self.verbose:
            print(f"   Applying PCA with {n_components*100}% variance retention...")
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        pca_time = time.time() - start_time
        if self.verbose:
            print(f" PCA completed in {pca_time:.2f}s")
            print(f"    Original dimensions: {X_train_dense.shape[1]}")
            print(f"    Reduced dimensions: {X_train_pca.shape[1]}")
            print(f"    Variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
            print(f"     Compression ratio: {X_train_pca.shape[1]/X_train_dense.shape[1]:.3f}")
        del X_train_dense, X_val_dense, X_test_dense, X_train_scaled, X_val_scaled, X_test_scaled
        gc.collect()
        self._log_memory_usage("After PCA")
        return X_train_pca, X_val_pca, X_test_pca
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 9: Model Training & Stacking")
            print("="*50)
        base_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=1000, random_state=42, n_jobs=self.n_jobs, max_depth=None,
                min_samples_split=2, min_samples_leaf=1, max_features='log2', bootstrap=True,
                class_weight='balanced_subsample', criterion='gini', oob_score=True,
                warm_start=False, max_samples=None
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=10000, n_jobs=self.n_jobs, solver='saga',
                C=1000.0, penalty='elasticnet', l1_ratio=0.5, class_weight='balanced',
                fit_intercept=True, tol=1e-6
            ),
            'SVM': SVC(
                random_state=42, kernel='rbf', C=1000.0, gamma='auto', probability=True,
                cache_size=5000, class_weight='balanced', shrinking=True, tol=1e-5, max_iter=10000
            )
        }
        results = {}
        individual_models = {}
        if self.verbose:
            print("\n Individual Model Training:")
        for name, model in base_models.items():
            if self.verbose:
                print(f"\n Training {name}...")
            start_time = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            train_time = time.time() - start_time
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'train_time': train_time,
                'predictions': y_pred
            }
            individual_models[name] = model
            if self.verbose:
                print(f"   {name}: {accuracy:.4f} accuracy ({train_time:.2f}s)")
        if self.verbose:
            print(f"\n Training Voting Ensemble...")
        start_time = time.time()
        estimators = [
            ('rf', base_models['Random Forest']),
            ('lr', base_models['Logistic Regression']),
            ('svm', base_models['SVM'])
        ]
        weights = [2, 5, 1]
        voting_classifier = VotingClassifier(
            estimators=estimators, voting='soft', weights=weights,
            n_jobs=self.n_jobs, flatten_transform=True
        )
        voting_classifier.fit(X_train, y_train)
        y_pred_voting = voting_classifier.predict(X_val)
        voting_accuracy = accuracy_score(y_val, y_pred_voting)
        voting_train_time = time.time() - start_time
        results['Voting Ensemble'] = {
            'model': voting_classifier,
            'accuracy': voting_accuracy,
            'train_time': voting_train_time,
            'predictions': y_pred_voting
        }
        if self.verbose:
            print(f"    Voting Ensemble: {voting_accuracy:.4f} accuracy ({voting_train_time:.2f}s)")
        if self.verbose and voting_accuracy >= 0.80:
            print(f"\n Voting Ensemble: {voting_accuracy:.4f} >= 0.80")
        self.models = {name: result['model'] for name, result in results.items()}
        self.results = results
        if self.verbose:
            print(f"\n Model Performance Summary:")
            print(f"     Random Forest: {results['Random Forest']['accuracy']:.4f}")
            print(f"     Logistic Regression: {results['Logistic Regression']['accuracy']:.4f}")
            print(f"     SVM: {results['SVM']['accuracy']:.4f}")
            print(f"     Voting Ensemble: {results['Voting Ensemble']['accuracy']:.4f}")
        return results
    
    def evaluate_and_visualize(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 10: Evaluation & Visualization")
            print("="*50)
        best_model_name = 'Voting Ensemble'
        best_predictions = self.results[best_model_name]['predictions']
        
        # Create visualizations
        self._create_visualizations()
        
        if self.verbose:
            print("  Evaluation completed")
            print("     Visualizations saved")
            print("\n  Detailed Classification Report (Voting Ensemble):")
            target_names = None
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                target_names = self.label_encoder.classes_
            print(classification_report(y_val, best_predictions, target_names=target_names))
    
    def save_trained_models(self) -> None:
        if self.verbose:
            print("\n" + "="*50)
            print("  SAVING TRAINED MODELS")
            print("="*50)
        models_dir = self.data_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        for model_name, model_data in self.results.items():
            model_file = models_dir / f'{model_name.lower().replace(" ", "_")}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model_data['model'], f)
            if self.verbose:
                print(f"    Saved: {model_name} -> {model_file.name}")
        preprocessing_components = {
            'vectorizer': self.vectorizer,
            'pca': self.pca,
            'feature_scaler': getattr(self, 'feature_scaler', None),
            'feature_selector': getattr(self, 'feature_selector', None),
            'label_encoder': getattr(self, 'label_encoder', None)
        }
        preprocessing_file = models_dir / 'preprocessing_components.pkl'
        with open(preprocessing_file, 'wb') as f:
            pickle.dump(preprocessing_components, f)
        if self.verbose:
            print(f"     Saved: Preprocessing Components -> {preprocessing_file.name}")
            print(f"     All models saved in: {models_dir}")
            print(f"     Total files saved: {len(list(models_dir.glob('*.pkl')))}")
    
    def _create_visualizations(self) -> None:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ML Pipeline Results Visualization', fontsize=16, fontweight='bold')
        
        # 1. Model Accuracy Comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = axes[0,0].bar(model_names, accuracies, color=colors[:len(model_names)])
        axes[0,0].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        for bar, acc in zip(bars, accuracies):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Training Time Comparison
        train_times = [self.results[name]['train_time'] for name in model_names]
        axes[0,1].bar(model_names, train_times, color=colors[:len(model_names)])
        axes[0,1].set_title('Training Time Comparison', fontweight='bold')
        axes[0,1].set_ylabel('Time (seconds)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. PCA Explained Variance
        if hasattr(self, 'pca') and self.pca is not None:
            cumsum_var = np.cumsum(self.pca.explained_variance_ratio_)
            axes[1,0].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2, markersize=6)
            axes[1,0].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
            axes[1,0].set_title('PCA Explained Variance Ratio', fontweight='bold')
            axes[1,0].set_xlabel('Principal Components')
            axes[1,0].set_ylabel('Cumulative Explained Variance')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Performance Summary Pie Chart
        best_acc = max(accuracies)
        performance_data = [best_acc, 1-best_acc]
        labels = ['Correct Predictions', 'Incorrect Predictions']
        colors_pie = ['#2ECC71', '#E74C3C']
        
        wedges, texts, autotexts = axes[1,1].pie(performance_data, labels=labels, colors=colors_pie, 
                                                 autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title(f'Best Model Performance\n({model_names[accuracies.index(best_acc)]})', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ml_pipeline_results.png', dpi=300, bbox_inches='tight')
        if self.verbose:
            print("     Visualization saved: ml_pipeline_results.png")
        plt.close()
    
    def _create_final_summary_plot(self, results: Dict[str, Any]) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create summary metrics
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(model_names))
        bars = ax.barh(y_pos, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(model_names)])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel('Accuracy Score')
        ax.set_title('Final Model Performance Summary', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add accuracy values on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{acc:.4f}', va='center', fontweight='bold')
        
        # Add target line
        ax.axvline(x=0.80, color='red', linestyle='--', alpha=0.7, label='Target (80%)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('final_model_summary.png', dpi=300, bbox_inches='tight')
        if self.verbose:
            print("     Final summary saved: final_model_summary.png")
        plt.close()
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        total_start_time = time.time()
        if self.verbose:
            print("  Starting Complete ML Pipeline")
            print("="*60)
        train_df, test_df, val_df = self.load_datasets()
        self.inspect_data(train_df, test_df, val_df)
        train_clean, test_clean, val_clean = self.clean_and_preprocess(train_df, test_df, val_df)
        train_encoded, test_encoded, val_encoded = self.encode_categorical_features(
            train_clean, test_clean, val_clean
        )
        X_train_tfidf, X_test_tfidf, X_val_tfidf = self.apply_tfidf_vectorization(
            train_encoded, test_encoded, val_encoded
        )
        X_train_split, X_val_split, y_train_split, y_val_split = self.split_data(
            X_train_tfidf, train_encoded
        )
        X_train_selected, X_val_selected, X_test_selected = self.apply_feature_selection(
            X_train_split, y_train_split, X_val_split, X_test_tfidf, k_features=8000
        )
        X_train_pca, X_val_pca, X_test_pca = self.apply_pca(
            X_train_selected, X_val_selected, X_test_selected, n_components=0.95
        )
        results = self.train_models(X_train_pca, y_train_split, X_val_pca, y_val_split)
        self.evaluate_and_visualize(X_val_pca, y_val_split)
        self.save_trained_models()
        total_time = time.time() - total_start_time
        if self.verbose:
            print("\n" + "="*60)
            print("  PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"   Total execution time: {total_time:.2f} seconds")
            voting_acc = results['Voting Ensemble']['accuracy']
            print(f"  Voting Ensemble: {voting_acc:.4f}")
            print(f"  Target accuracy range: 0.80 - 0.92")
            if voting_acc >= 0.80:
                print(f"  TARGET ACCURACY ACHIEVED! ({voting_acc:.4f} >= 0.80)")
            else:
                print(f"   Target accuracy not reached ({voting_acc:.4f} < 0.80)")
            print(f"  Final memory usage: {psutil.virtual_memory().percent:.1f}%")
            print(f"  Visualizations: ml_pipeline_results.png, final_model_summary.png")
        # Generate final visualization summary
        if self.verbose:
            self._create_final_summary_plot(results)
        
        return {
            'results': results,
            'total_time': total_time,
            'vectorizer': self.vectorizer,
            'pca': self.pca,
            'models': self.models
        }


def main():
    pipeline = OptimizedMLPipeline(data_dir=".", n_jobs=-1, verbose=True)
    results = pipeline.run_complete_pipeline()
    return results


if __name__ == "__main__":
    results = main()
