# 🚀 Optimized ML Pipeline for Medical Question Classification

A high-performance machine learning pipeline designed to train and evaluate models on medical question datasets with maximum efficiency and speed.

## 📋 Features

- **Complete 10-Step ML Pipeline**: Follows industry best practices
- **Performance Optimized**: Multi-core processing, memory management, and fast algorithms
- **Multiple Models**: Compares Logistic Regression, Random Forest, Naive Bayes, and Gradient Boosting
- **Rich Visualizations**: Interactive dashboard with model comparisons and results
- **Memory Efficient**: Optimized data types and garbage collection
- **Progress Tracking**: Real-time progress bars and performance metrics

## 🔧 Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure your datasets are in the project directory**:
   - `train.csv`
   - `test.csv` 
   - `validation.csv`

## 📊 Dataset Format

Your CSV files should contain these columns:
- `id`: Unique identifier
- `question`: The main question text
- `opa`, `opb`, `opc`, `opd`: Multiple choice options
- `cop`: Correct option (integer)
- `choice_type`: Type of choice (single/multi)
- `exp`: Explanation text
- `subject_name`: Subject category (target variable)
- `topic_name`: Topic category

## 🚀 Usage

### Quick Start
```python
python ml_pipeline.py
```

### Advanced Usage
```python
from ml_pipeline import OptimizedMLPipeline

# Initialize pipeline
pipeline = OptimizedMLPipeline(
    data_dir=".",           # Directory containing CSV files
    n_jobs=-1,              # Use all CPU cores (-1) or specify number
    verbose=True            # Enable detailed logging
)

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Access results
best_model = max(results['results'].keys(), 
                key=lambda k: results['results'][k]['accuracy'])
print(f"Best model: {best_model}")
print(f"Accuracy: {results['results'][best_model]['accuracy']:.4f}")
```

## 📈 Pipeline Steps

### 1. **Import Libraries**
- Optimized imports with performance-focused libraries
- Automatic NLTK data downloading

### 2. **Load Datasets**
- Memory-efficient data loading with optimized dtypes
- Automatic memory usage tracking

### 3. **Inspect Data**
- Comprehensive data analysis
- Missing value detection
- Duplicate identification
- Memory usage reporting

### 4. **Clean Text & Handle Missing Values**
- Parallel text cleaning with Numba optimization
- Smart missing value imputation
- Duplicate removal
- Special character handling

### 5. **Encode Categorical Features**
- Label encoding for categorical variables
- Consistent encoding across train/test/validation sets
- Memory-efficient processing

### 6. **TF-IDF Vectorization**
- Optimized TF-IDF parameters for speed
- Combined text features (question + options)
- Sparse matrix handling
- Vocabulary size optimization

### 7. **Train/Test Split**
- Stratified splitting for balanced classes
- Additional validation split for model selection
- Target variable encoding

### 8. **PCA Dimensionality Reduction**
- Configurable variance retention (default: 95%)
- Memory-efficient dense matrix conversion
- Compression ratio reporting

### 9. **Train ML Models**
- **Logistic Regression**: Fast linear classifier
- **Random Forest**: Ensemble method with parallel processing
- **Naive Bayes**: Probabilistic classifier
- **Gradient Boosting**: Advanced ensemble method
- Parallel model training
- Cross-validation support

### 10. **Evaluate & Visualize**
- Interactive HTML dashboard
- Model accuracy comparison
- Training time analysis
- Confusion matrices
- Feature importance plots
- Detailed classification reports

## ⚡ Performance Optimizations

### Speed Enhancements
- **Multi-core Processing**: Utilizes all available CPU cores
- **Numba JIT Compilation**: Fast text cleaning operations
- **Optimized Algorithms**: Fast solvers and reduced complexity
- **Sparse Matrix Operations**: Memory-efficient TF-IDF handling
- **Parallel Model Training**: Simultaneous model evaluation

### Memory Management
- **Optimized Data Types**: Reduced memory footprint
- **Garbage Collection**: Automatic memory cleanup
- **Memory Monitoring**: Real-time usage tracking
- **Efficient Data Structures**: Sparse matrices and categorical types

### Algorithm Optimizations
- **Limited Feature Sets**: Balanced performance vs. speed
- **Reduced Model Complexity**: Optimized hyperparameters
- **Smart Preprocessing**: Efficient text cleaning pipeline
- **Vectorization Limits**: Controlled vocabulary size

## 📊 Output Files

After running the pipeline, you'll get:

1. **`ml_results_dashboard.html`**: Interactive visualization dashboard
2. **Console Output**: Detailed progress and performance metrics
3. **Model Objects**: Trained models stored in pipeline object

## 🎯 Expected Performance

### Typical Results
- **Training Time**: 30-120 seconds (depending on dataset size)
- **Memory Usage**: 2-8GB RAM (depending on dataset size)
- **Accuracy**: 70-95% (depending on data quality)
- **Feature Reduction**: 80-95% dimensionality reduction

### Hardware Recommendations
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB+ recommended for large datasets
- **Storage**: SSD for faster I/O operations

## 🔧 Customization

### Modify TF-IDF Parameters
```python
# In the apply_tfidf_vectorization method
self.vectorizer = TfidfVectorizer(
    max_features=10000,     # Increase for more features
    min_df=3,               # Adjust minimum document frequency
    max_df=0.9,             # Adjust maximum document frequency
    ngram_range=(1, 3)      # Include trigrams
)
```

### Adjust PCA Components
```python
# In the apply_pca method call
X_train_pca, X_val_pca, X_test_pca = self.apply_pca(
    X_train_split, X_val_split, X_test_tfidf,
    n_components=0.99  # Retain 99% variance
)
```

### Add Custom Models
```python
# In the train_models method
models['SVM'] = SVC(
    kernel='rbf',
    random_state=42,
    probability=True
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**
   - Reduce `max_features` in TF-IDF
   - Increase system RAM
   - Process datasets in smaller batches

2. **Slow Performance**
   - Reduce `n_estimators` in ensemble models
   - Limit PCA components
   - Use fewer CPU cores if system is overloaded

3. **Import Errors**
   - Ensure all requirements are installed
   - Check Python version compatibility (3.8+)

4. **NLTK Data Missing**
   - The pipeline automatically downloads required NLTK data
   - Manual download: `nltk.download('punkt')`, `nltk.download('stopwords')`

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the pipeline.

## 📞 Support

For questions or issues, please check the troubleshooting section or create an issue in the project repository.

---

**Happy Machine Learning! 🚀**
