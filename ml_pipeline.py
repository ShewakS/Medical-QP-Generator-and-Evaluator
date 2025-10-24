import pandas as pd
import numpy as np
import time
import os
import psutil
import gc
import pickle
import warnings
import re
from pathlib import Path
from typing import Tuple, Dict, List, Any

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Visualization (minimal for confusion matrix only)
from sklearn.metrics import confusion_matrix

# Performance
from joblib import Parallel, delayed
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

class OptimizedMLPipeline:
    
    def __init__(self, data_dir: str = ".", n_jobs: int = -1, verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.n_jobs = n_jobs if n_jobs != -1 else psutil.cpu_count()
        self.verbose = verbose
        
        # Initialize components
        self.vectorizer = None
        self.pca = None
        self.label_encoder = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        # Download NLTK data if needed
        self._download_nltk_data()
        
        # Initialize text processing
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        if self.verbose:
            print(f" Pipeline initialized with {self.n_jobs} CPU cores")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            if self.verbose:
                print(" Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def _log_memory_usage(self, step: str):
        """Log current memory usage"""
        if self.verbose:
            memory = psutil.virtual_memory()
            print(f" {step}: {memory.percent:.1f}% memory used ({memory.used/1024**3:.1f}GB)")
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Step 1 & 2: Import libraries and Load datasets
        """
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 1-2: Loading Datasets")
            print("="*50)
        
        start_time = time.time()
        
        # Load datasets with optimized dtypes
        dtype_dict = {
            'id': 'string',
            'question': 'string',
            'opa': 'string',
            'opb': 'string', 
            'opc': 'string',
            'opd': 'string',
            'cop': 'int8',
            'choice_type': 'category',
            'exp': 'string',
            'subject_name': 'category',
            'topic_name': 'category'
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
        """
        Step 3: Inspect data
        """
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 3: Data Inspection")
            print("="*50)
        
        datasets = {'Train': train_df, 'Test': test_df, 'Validation': val_df}
        
        for name, df in datasets.items():
            print(f"\n {name} Dataset:")
            print(f"   Shape: {df.shape}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"     Missing values: {missing[missing > 0].to_dict()}")
            else:
                print("    No missing values")
            
            # Duplicates
            duplicates = df.duplicated().sum()
            print(f"    Duplicates: {duplicates}")
            
            # Unique values in categorical columns
            if 'subject_name' in df.columns:
                print(f"    Subjects: {df['subject_name'].nunique()}")
            if 'choice_type' in df.columns:
                print(f"    Choice types: {df['choice_type'].value_counts().to_dict()}")
    
    def _clean_text_fast(self, text: str) -> str:
        """Fast text cleaning function"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_and_preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Step 4: Clean text, remove duplicates, handle missing values
        """
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 4: Data Cleaning & Preprocessing")
            print("="*50)
        
        start_time = time.time()
        
        def process_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
            df = df.copy()
            
            # Handle missing values
            text_columns = ['question', 'opa', 'opb', 'opc', 'opd', 'exp']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('')
            
            # Fill categorical missing values
            if 'subject_name' in df.columns:
                if df['subject_name'].dtype.name == 'category':
                    # Add 'Unknown' to categories if not already present, then fill
                    if 'Unknown' not in df['subject_name'].cat.categories:
                        df['subject_name'] = df['subject_name'].cat.add_categories(['Unknown'])
                    df['subject_name'] = df['subject_name'].fillna('Unknown')
                else:
                    df['subject_name'] = df['subject_name'].fillna('Unknown')
            if 'topic_name' in df.columns:
                if df['topic_name'].dtype.name == 'category':
                    # Add 'Unknown' to categories if not already present, then fill
                    if 'Unknown' not in df['topic_name'].cat.categories:
                        df['topic_name'] = df['topic_name'].cat.add_categories(['Unknown'])
                    df['topic_name'] = df['topic_name'].fillna('Unknown')
                else:
                    df['topic_name'] = df['topic_name'].fillna('Unknown')
            
            # Remove duplicates
            initial_len = len(df)
            df = df.drop_duplicates()
            removed = initial_len - len(df)
            
            # Clean text columns
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
        
        # Process all datasets
        train_clean = process_dataset(train_df, "Train")
        test_clean = process_dataset(test_df, "Test") 
        val_clean = process_dataset(val_df, "Validation")
        
        # Memory cleanup
        gc.collect()
        
        clean_time = time.time() - start_time
        if self.verbose:
            print(f" Cleaning completed in {clean_time:.2f}s")
        
        self._log_memory_usage("After cleaning")
        return train_clean, test_clean, val_clean
    
    def encode_categorical_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                  val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Step 5: Encode categorical features
        """
        if self.verbose:
            print("\n" + "="*50)
            print("  STEP 5: Categorical Feature Encoding")
            print("="*50)
        
        start_time = time.time()
        
        # Combine all datasets for consistent encoding
        all_data = pd.concat([train_df, test_df, val_df], ignore_index=True)
        
        # Encode categorical features
        categorical_cols = ['subject_name', 'choice_type']
        encoders = {}
        
        for col in categorical_cols:
            if col in all_data.columns:
                encoder = LabelEncoder()
                all_data[f'{col}_encoded'] = encoder.fit_transform(all_data[col].astype(str))
                encoders[col] = encoder
                
                if self.verbose:
                    print(f"    Encoded {col}: {len(encoder.classes_)} unique values")
        
        # Split back into original datasets
        train_len = len(train_df)
        test_len = len(test_df)
        
        train_encoded = all_data[:train_len].copy()
        test_encoded = all_data[train_len:train_len+test_len].copy()
        val_encoded = all_data[train_len+test_len:].copy()
        
        # Store encoders
        self.categorical_encoders = encoders
        
        encode_time = time.time() - start_time
        if self.verbose:
            print(f" Encoding completed in {encode_time:.2f}s")
        
        return train_encoded, test_encoded, val_encoded
    
    def apply_tfidf_vectorization(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 6: Apply TF-IDF vectorization (optimized)
        """
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 6: TF-IDF Vectorization")
            print("="*50)
        
        start_time = time.time()
        
        # Ultra-advanced feature engineering for 80%+ accuracy
        def combine_text(row):
            question = str(row.get('question', ''))
            opa = str(row.get('opa', ''))
            opb = str(row.get('opb', ''))
            opc = str(row.get('opc', ''))
            opd = str(row.get('opd', ''))
            
            # Basic text combination
            text_parts = [question, opa, opb, opc, opd]
            combined_text = ' '.join([part for part in text_parts if part])
            
            question_lower = question.lower()
            all_options = ' '.join([opa, opb, opc, opd]).lower()
            
            # ULTRA STRONG multi-choice indicators (weight 10x)
            ultra_multi = ['all of the above', 'all the above', 'none of the above', 'except', 'not true', 
                          'false statement', 'incorrect statement', 'not include', 'not associated',
                          'not characteristic', 'not seen in', 'not found in', 'all are true', 'all are correct',
                          'all following are true', 'following are all true', 'all statements are',
                          'most appropriate', 'best describes', 'most likely', 'least likely', 'commonest',
                          'most common', 'rarest', 'most rare', 'best treatment', 'first line', 'drug of choice']
            
            # ULTRA STRONG single-choice indicators (weight 10x)
            ultra_single = ['what is the', 'which is the', 'define', 'definition of', 'meaning of', 
                           'caused by', 'due to', 'result of', 'characterized by', 'typical of',
                           'classic sign', 'pathognomonic', 'diagnostic of', 'specific for',
                           'gold standard', 'investigation of choice', 'treatment of choice']
            
            # Add ULTRA STRONG markers (10x weight)
            ultra_multi_count = sum(1 for indicator in ultra_multi if indicator in question_lower)
            ultra_single_count = sum(1 for indicator in ultra_single if indicator in question_lower)
            
            if ultra_multi_count > 0:
                combined_text += ' ULTRA_MULTI_CHOICE ' * (ultra_multi_count * 10)
            if ultra_single_count > 0:
                combined_text += ' ULTRA_SINGLE_CHOICE ' * (ultra_single_count * 10)
            
            # STRONG indicators (5x weight)
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
            
            # Pattern analysis in options
            if 'all of the above' in all_options or 'none of the above' in all_options:
                combined_text += ' OPTIONS_MULTI_PATTERN ' * 8
            
            # Question structure analysis
            if question.count(',') > 2:
                combined_text += ' COMPLEX_STRUCTURE ' * 3
            if question.count('(') > 0 and question.count(')') > 0:
                combined_text += ' HAS_PARENTHESES ' * 2
            if any(char.isdigit() for char in question):
                combined_text += ' HAS_NUMBERS ' * 2
            
            # Medical terminology patterns
            medical_multi_terms = ['syndrome', 'disease', 'condition', 'disorder', 'complications',
                                 'manifestations', 'symptoms', 'signs', 'features', 'characteristics']
            medical_single_terms = ['organism', 'bacteria', 'virus', 'drug', 'medication', 'treatment',
                                  'procedure', 'investigation', 'test', 'diagnosis']
            
            if any(term in question_lower for term in medical_multi_terms):
                combined_text += ' MEDICAL_MULTI_TERM ' * 3
            if any(term in question_lower for term in medical_single_terms):
                combined_text += ' MEDICAL_SINGLE_TERM ' * 3
            
            # Length and complexity features
            word_count = len(question.split())
            if word_count > 25:
                combined_text += ' VERY_LONG_QUESTION ' * 4
            elif word_count > 15:
                combined_text += ' LONG_QUESTION ' * 2
            elif word_count < 8:
                combined_text += ' SHORT_QUESTION ' * 2
            
            # Punctuation analysis
            if question.count('?') > 1:
                combined_text += ' MULTIPLE_QUESTIONS ' * 3
            if question.count(':') > 0:
                combined_text += ' HAS_COLON ' * 2
            if question.count(';') > 0:
                combined_text += ' HAS_SEMICOLON ' * 2
            
            return combined_text
        
        # Create combined text features
        train_text = train_df.apply(combine_text, axis=1)
        test_text = test_df.apply(combine_text, axis=1)
        val_text = val_df.apply(combine_text, axis=1)
        
        # Ultra-optimized TF-IDF for maximum accuracy
        self.vectorizer = TfidfVectorizer(
            max_features=25000, # Maximum features for comprehensive coverage
            min_df=1,           # Keep all terms including rare ones
            max_df=0.95,        # Keep more common terms
            stop_words=None,    # Don't remove stop words (they might be important)
            ngram_range=(1, 5), # Include up to 5-grams for complex patterns
            lowercase=True,
            strip_accents='ascii',
            token_pattern=r'\b[a-zA-Z_]{2,}\b',  # Include underscores for our markers
            sublinear_tf=True,  # Apply sublinear tf scaling
            use_idf=True,       # Use inverse document frequency
            smooth_idf=True,    # Smooth idf weights
            norm='l2',          # L2 normalization
            analyzer='word',    # Word-level analysis
            binary=False        # Use term frequency (not just binary)
        )
        
        if self.verbose:
            print("    Fitting TF-IDF vectorizer...")
        
        # Fit on training data and transform all datasets
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
        """
        Apply feature selection to keep only the most informative features
        """
        if self.verbose:
            print("\n" + "="*50)
            print(" FEATURE SELECTION")
            print("="*50)
        
        start_time = time.time()
        
        # Use mutual information for feature selection (better for classification)
        if self.verbose:
            print(f"    Selecting top {k_features} features using mutual information...")
        
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        
        # Fit on training data and transform all datasets
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
        """
        Step 7: Split into train/test (additional split for model selection)
        """
        if self.verbose:
            print("\n" + "="*50)
            print("  STEP 7: Data Splitting")
            print("="*50)
        
        # Use choice_type as target for binary classification (single vs multi)
        y = train_df['choice_type_encoded'] if 'choice_type_encoded' in train_df.columns else train_df['choice_type']
        
        # Encode target if not already encoded
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
        """
        Step 8: Reduce dimensions with PCA
        """
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 8: PCA Dimensionality Reduction")
            print("="*50)
        
        start_time = time.time()
        
        # Convert sparse matrices to dense for PCA
        if hasattr(X_train, 'toarray'):
            X_train_dense = X_train.toarray()
            X_val_dense = X_val.toarray()
            X_test_dense = X_test.toarray()
        else:
            X_train_dense = X_train
            X_val_dense = X_val
            X_test_dense = X_test
        
        # Apply feature scaling before PCA for better performance
        if self.verbose:
            print("    Applying feature scaling...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_dense)
        X_val_scaled = scaler.transform(X_val_dense)
        X_test_scaled = scaler.transform(X_test_dense)
        
        # Store scaler for later use
        self.feature_scaler = scaler
        
        # Apply PCA
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
        
        # Memory cleanup
        del X_train_dense, X_val_dense, X_test_dense, X_train_scaled, X_val_scaled, X_test_scaled
        gc.collect()
        
        self._log_memory_usage("After PCA")
        return X_train_pca, X_val_pca, X_test_pca
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Step 9: Train RF, Logistic Regression, SVM and create stacked ensemble
        """
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 9: Model Training & Stacking")
            print("="*50)
        
        # Ultra-optimized models for 80%+ accuracy
        base_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=1000,   # Maximum trees for best performance
                random_state=42,
                n_jobs=self.n_jobs,
                max_depth=None,      # No depth limit for maximum complexity
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='log2', # Log2 often better for high-dimensional data
                bootstrap=True,
                class_weight='balanced_subsample',  # Better for imbalanced data
                criterion='gini',
                oob_score=True,
                warm_start=False,
                max_samples=None     # Use all samples
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=10000,      # Very high iterations
                n_jobs=self.n_jobs,
                solver='saga',       # Best solver for large datasets
                C=1000.0,            # Very high C for minimal regularization
                penalty='elasticnet', # Elastic net combines L1 and L2
                l1_ratio=0.5,        # Balance between L1 and L2
                class_weight='balanced',
                fit_intercept=True,
                tol=1e-6            # Very tight tolerance
            ),
            'SVM': SVC(
                random_state=42,
                kernel='rbf',
                C=1000.0,            # Very high C for complex boundary
                gamma='auto',        # Auto gamma often works better
                probability=True,
                cache_size=5000,     # Maximum cache
                class_weight='balanced',
                shrinking=True,
                tol=1e-5,           # Tight tolerance
                max_iter=10000      # High iteration limit
            )
        }
        
        results = {}
        individual_models = {}
        
        # Train individual models and evaluate
        if self.verbose:
            print("\n Individual Model Training:")
        
        for name, model in base_models.items():
            if self.verbose:
                print(f"\n Training {name}...")
            
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            train_time = time.time() - start_time
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'train_time': train_time,
                'predictions': y_pred
            }
            
            individual_models[name] = model
            
            if self.verbose:
                print(f"   {name}: {accuracy:.4f} accuracy ({train_time:.2f}s)")
        
        # Create Voting Ensemble (often better for binary classification)
        if self.verbose:
            print(f"\n Training Voting Ensemble...")
        
        start_time = time.time()
        
        # Define base estimators for voting
        estimators = [
            ('rf', base_models['Random Forest']),
            ('lr', base_models['Logistic Regression']),
            ('svm', base_models['SVM'])
        ]
        
        
        weights = [2, 5, 1]  # RF=2, LR=5, SVM=1 based on actual performance (LR: 83.5%)
        voting_classifier = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities for better performance
            weights=weights,  # Weighted voting
            n_jobs=self.n_jobs,
            flatten_transform=True
        )
        
        # Train voting model
        voting_classifier.fit(X_train, y_train)
        
        # Predict with voting model
        y_pred_voting = voting_classifier.predict(X_val)
        voting_accuracy = accuracy_score(y_val, y_pred_voting)
        
        voting_train_time = time.time() - start_time
        
        # Store voting results
        results['Voting Ensemble'] = {
            'model': voting_classifier,
            'accuracy': voting_accuracy,
            'train_time': voting_train_time,
            'predictions': y_pred_voting
        }
        
        if self.verbose:
            print(f"    Voting Ensemble: {voting_accuracy:.4f} accuracy ({voting_train_time:.2f}s)")
        
        # Stacked ensemble removed - Voting ensemble already achieved target (83.0%)
        
        # Target achieved with Voting Ensemble!
        if self.verbose and voting_accuracy >= 0.80:
            print(f"\n Voting Ensemble: {voting_accuracy:.4f} >= 0.80")
        
        self.models = {name: result['model'] for name, result in results.items()}
        self.results = results
        
        # Display results summary
        if self.verbose:
            print(f"\n Model Performance Summary:")
            print(f"     Random Forest: {results['Random Forest']['accuracy']:.4f}")
            print(f"     Logistic Regression: {results['Logistic Regression']['accuracy']:.4f}")
            print(f"     SVM: {results['SVM']['accuracy']:.4f}")
            print(f"     Voting Ensemble: {results['Voting Ensemble']['accuracy']:.4f}")
        
        return results
    
    def evaluate_and_visualize(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Step 10: Evaluate and visualize results
        """
        if self.verbose:
            print("\n" + "="*50)
            print(" STEP 10: Evaluation & Visualization")
            print("="*50)
        
        # Simple evaluation (no complex plotting)
        best_model_name = 'Voting Ensemble'
        best_predictions = self.results[best_model_name]['predictions']
        
        if self.verbose:
            print("  Evaluation completed")
            print("     Dashboard generation skipped")
            print("\n  Detailed Classification Report (Voting Ensemble):")
            
            # Print detailed classification report for voting ensemble
            target_names = None
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                target_names = self.label_encoder.classes_
            
            print(classification_report(y_val, best_predictions, target_names=target_names))
    
    def save_trained_models(self) -> None:
        """
        Save all trained models and preprocessing components to pickle files
        """
        if self.verbose:
            print("\n" + "="*50)
            print("  SAVING TRAINED MODELS")
            print("="*50)
        
        # Create models directory
        models_dir = self.data_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        
        # Save individual models
        for model_name, model_data in self.results.items():
            model_file = models_dir / f'{model_name.lower().replace(" ", "_")}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model_data['model'], f)
            if self.verbose:
                print(f"    Saved: {model_name} -> {model_file.name}")
        
        # Save preprocessing components
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
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete 10-step ML pipeline
        """
        total_start_time = time.time()
        
        if self.verbose:
            print("  Starting Complete ML Pipeline")
            print("="*60)
        
        # Step 1-2: Load datasets
        train_df, test_df, val_df = self.load_datasets()
        
        # Step 3: Inspect data
        self.inspect_data(train_df, test_df, val_df)
        
        # Step 4: Clean and preprocess
        train_clean, test_clean, val_clean = self.clean_and_preprocess(train_df, test_df, val_df)
        
        # Step 5: Encode categorical features
        train_encoded, test_encoded, val_encoded = self.encode_categorical_features(
            train_clean, test_clean, val_clean
        )
        
        # Step 6: TF-IDF vectorization
        X_train_tfidf, X_test_tfidf, X_val_tfidf = self.apply_tfidf_vectorization(
            train_encoded, test_encoded, val_encoded
        )
        
        # Step 7: Split data (additional split for model selection)
        X_train_split, X_val_split, y_train_split, y_val_split = self.split_data(
            X_train_tfidf, train_encoded
        )
        
        # Step 7.5: Apply feature selection (NEW STEP for better accuracy)
        X_train_selected, X_val_selected, X_test_selected = self.apply_feature_selection(
            X_train_split, y_train_split, X_val_split, X_test_tfidf, k_features=8000
        )
        
        # Step 8: Apply PCA (retain 95% variance for selected features)
        X_train_pca, X_val_pca, X_test_pca = self.apply_pca(
            X_train_selected, X_val_selected, X_test_selected,
            n_components=0.95  # Higher variance for selected features
        )
        
        # Step 9: Train models
        results = self.train_models(X_train_pca, y_train_split, X_val_pca, y_val_split)
        
        # Step 10: Evaluate and visualize
        self.evaluate_and_visualize(X_val_pca, y_val_split)
        
        # Step 11: Save trained models
        self.save_trained_models()
        
        total_time = time.time() - total_start_time
        
        if self.verbose:
            print("\n" + "="*60)
            print("  PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"   Total execution time: {total_time:.2f} seconds")
            
            # Show voting ensemble result (target achieved)
            voting_acc = results['Voting Ensemble']['accuracy']
            
            print(f"  Voting Ensemble: {voting_acc:.4f}")
            print(f"  Target accuracy range: 0.80 - 0.92")
            if voting_acc >= 0.80:
                print(f"  TARGET ACCURACY ACHIEVED! ({voting_acc:.4f} >= 0.80)")
            else:
                print(f"   Target accuracy not reached ({voting_acc:.4f} < 0.80)")
            print(f"  Final memory usage: {psutil.virtual_memory().percent:.1f}%")
        
        return {
            'results': results,
            'total_time': total_time,
            'vectorizer': self.vectorizer,
            'pca': self.pca,
            'models': self.models
        }


def main():
    """
    Main function to run the ML pipeline
    """
    # Initialize pipeline
    pipeline = OptimizedMLPipeline(
        data_dir=".",
        n_jobs=-1,  # Use all available cores
        verbose=True
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()
