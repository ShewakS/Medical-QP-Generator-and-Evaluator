# 🏥 Medical Question Generator & Evaluator - Frontend

A modern, interactive web application for generating and evaluating medical questions using AI-powered machine learning models.

## 🌟 Features

### 📚 **Question Generation**
- **15 Medical Topics**: Anatomy, Physiology, Pathology, Pharmacology, Microbiology, Biochemistry, and more
- **Flexible Question Count**: Generate 1-20 questions per session
- **3 Difficulty Levels**: Easy, Medium, Hard
- **AI-Powered**: Uses trained ML models for intelligent question generation

### 🎯 **Interactive Testing**
- **Modern UI**: Clean, responsive design with smooth animations
- **Real-time Timer**: Track your test duration
- **Progress Tracking**: Visual progress bar and question counter
- **Easy Navigation**: Previous/Next buttons with smart validation

### 📊 **Comprehensive Evaluation**
- **Detailed Scoring**: Percentage score with performance analysis
- **Question Review**: See correct answers and explanations
- **Performance Insights**: Personalized feedback based on your score
- **Results Export**: Download your results as JSON

## 🚀 Quick Start

### Option 1: Run with Flask API (Recommended)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_api.txt
   ```

2. **Start the Application**:
   ```bash
   python run_app.py
   ```

3. **Open Browser**:
   Navigate to `http://localhost:5000`

### Option 2: Standalone HTML (No Backend)

1. **Open HTML File**:
   Simply open `templates/index.html` in your browser
   
2. **Note**: Will use mock data instead of AI-generated questions

## 📁 Project Structure

```
ml project/
├── templates/
│   └── index.html          # Main HTML interface
├── static/
│   ├── styles.css          # Modern CSS styling
│   └── script.js           # Interactive JavaScript
├── trained_models/         # ML models (from ml_pipeline.py)
├── api.py                  # Flask backend API
├── run_app.py             # Application launcher
├── requirements_api.txt    # Frontend dependencies
└── README_FRONTEND.md     # This file
```

## 🎨 User Interface

### 1. **Question Generation Screen**
- Select medical topic from dropdown
- Choose number of questions (1-20)
- Pick difficulty level
- Click "Generate Questions" button

### 2. **Testing Interface**
- Clean question display with multiple choice options
- Real-time timer in the header
- Progress bar showing completion status
- Navigation buttons (Previous/Next/Submit)

### 3. **Results Dashboard**
- Circular score display with color coding:
  - 🟢 Green: 90%+ (Excellent)
  - 🟡 Yellow: 80-89% (Good)
  - 🔴 Red: Below 80% (Needs Improvement)
- Detailed statistics (Correct/Incorrect/Time)
- Performance analysis with personalized feedback
- Question-by-question review with explanations

## 🔧 Technical Features

### **Frontend Technologies**
- **HTML5**: Semantic markup with accessibility features
- **CSS3**: Modern styling with gradients, animations, and responsive design
- **JavaScript ES6+**: Async/await, classes, and modern syntax
- **Font Awesome**: Professional icons throughout the interface

### **Backend Integration**
- **Flask API**: RESTful endpoints for question generation and evaluation
- **ML Integration**: Uses trained models (83% accuracy) for intelligent questions
- **Fallback System**: Works with mock data if API is unavailable

### **Responsive Design**
- **Mobile-First**: Optimized for all screen sizes
- **Touch-Friendly**: Large buttons and touch targets
- **Cross-Browser**: Compatible with modern browsers

## 🎯 API Endpoints

### `POST /api/generate-questions`
Generate questions based on topic and difficulty
```json
{
  "topic": "anatomy",
  "count": 5,
  "difficulty": "medium"
}
```

### `POST /api/evaluate`
Evaluate user answers and provide detailed feedback
```json
{
  "questions": [...],
  "user_answers": {...}
}
```

### `GET /api/topics`
Get list of available medical topics

### `GET /api/health`
Check API and model status

## 📊 Performance Features

### **Smart Question Generation**
- Uses trained ML models with 83% accuracy
- Topic-specific question templates
- Difficulty-based question complexity
- Randomized option ordering

### **Intelligent Evaluation**
- Real-time answer validation
- Detailed performance analysis
- Personalized feedback based on score
- Comprehensive question review

### **User Experience**
- Loading animations during question generation
- Smooth transitions between sections
- Auto-save of user progress
- Keyboard navigation support

## 🎨 Design Highlights

### **Modern UI Elements**
- Gradient backgrounds and buttons
- Card-based layout with shadows
- Smooth hover effects and transitions
- Professional color scheme

### **Interactive Components**
- Animated loading spinner
- Progress indicators
- Color-coded results
- Responsive navigation

### **Accessibility Features**
- Semantic HTML structure
- Keyboard navigation
- Screen reader friendly
- High contrast colors

## 🔍 Usage Examples

### **For Students**
1. Select "Anatomy" topic
2. Choose 10 questions, Medium difficulty
3. Take the test with timer running
4. Review results and explanations
5. Download results for study tracking

### **For Educators**
1. Generate questions for different topics
2. Use various difficulty levels for assessment
3. Review student performance patterns
4. Export results for grading

### **For Self-Assessment**
1. Test knowledge across multiple topics
2. Track improvement over time
3. Focus on weak areas using explanations
4. Challenge yourself with harder difficulties

## 🚀 Advanced Features

### **Question Bank Integration**
- Connects to trained ML models
- Dynamic question generation
- Topic-specific content
- Difficulty scaling

### **Results Analytics**
- Performance tracking
- Time analysis
- Topic-wise scoring
- Improvement suggestions

### **Export Capabilities**
- JSON format results
- Detailed question breakdown
- Performance metrics
- Timestamp tracking

## 🛠️ Customization

### **Adding New Topics**
Edit `api.py` to add new medical topics in the `question_templates` dictionary.

### **Modifying Difficulty Levels**
Adjust question complexity in the template variations.

### **Styling Changes**
Modify `static/styles.css` for custom themes and colors.

### **Functionality Extensions**
Extend `static/script.js` for additional features.

## 📱 Browser Compatibility

- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+
- ✅ Mobile browsers

## 🎉 Success Metrics

- **83% ML Model Accuracy**: Powered by optimized ensemble models
- **Responsive Design**: Works on all devices
- **Fast Performance**: Optimized loading and interactions
- **User-Friendly**: Intuitive interface with clear navigation

## 🔮 Future Enhancements

- User authentication and progress tracking
- Advanced analytics dashboard
- Question difficulty adaptation
- Social features and leaderboards
- Offline mode support
- Multi-language support

---

**🏥 Ready to test your medical knowledge? Start the application and begin your learning journey!**
