// Medical Question Generator & Evaluator
class MedicalQuestionApp {
    constructor() {
        this.questions = [];
        this.currentQuestionIndex = 0;
        this.userAnswers = {};
        this.startTime = null;
        this.endTime = null;
        this.timer = null;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.showSection('generation-section');
    }

    bindEvents() {
        // Form submission
        document.getElementById('question-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.generateQuestions();
        });

        // Navigation buttons
        document.getElementById('prev-btn').addEventListener('click', () => {
            this.navigateQuestion(-1);
        });

        document.getElementById('next-btn').addEventListener('click', () => {
            this.navigateQuestion(1);
        });

        document.getElementById('submit-btn').addEventListener('click', () => {
            this.submitTest();
        });

        // Restart button
        document.getElementById('restart-btn').addEventListener('click', () => {
            this.restart();
        });

        // Download results
        document.getElementById('download-results').addEventListener('click', () => {
            this.downloadResults();
        });
    }

    showSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        
        // Show target section
        document.getElementById(sectionId).classList.add('active');
    }

    showLoading(show = true) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.add('active');
        } else {
            overlay.classList.remove('active');
        }
    }

    async generateQuestions() {
        const topic = document.getElementById('topic-select').value;
        const count = parseInt(document.getElementById('question-count').value);
        const difficulty = document.getElementById('difficulty-level').value;

        if (!topic || !count) {
            alert('Please select a topic and number of questions.');
            return;
        }

        this.showLoading(true);

        try {
            // Try to use Flask API first, fallback to mock data
            let questions;
            try {
                const response = await fetch('/api/generate-questions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        topic: topic,
                        count: count,
                        difficulty: difficulty
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    questions = data.questions;
                    console.log('✅ Questions generated via API');
                } else {
                    throw new Error('API request failed');
                }
            } catch (apiError) {
                console.log('⚠️ API not available, using mock data');
                questions = this.generateMockQuestions(topic, count, difficulty);
            }
            
            this.questions = questions;
            this.currentQuestionIndex = 0;
            this.userAnswers = {};
            
            this.displayQuestions();
            this.showSection('questions-section');
            this.startTimer();
            
        } catch (error) {
            console.error('Error generating questions:', error);
            alert('Error generating questions. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }

    generateMockQuestions(topic, count, difficulty) {
        const questionBank = {
            anatomy: [
                {
                    question: "Which bone is the longest in the human body?",
                    options: ["Tibia", "Femur", "Humerus", "Radius"],
                    correct: 1,
                    explanation: "The femur (thigh bone) is the longest and strongest bone in the human body."
                },
                {
                    question: "How many chambers does a human heart have?",
                    options: ["2", "3", "4", "5"],
                    correct: 2,
                    explanation: "The human heart has four chambers: two atria and two ventricles."
                },
                {
                    question: "Which part of the brain controls balance and coordination?",
                    options: ["Cerebrum", "Cerebellum", "Medulla", "Pons"],
                    correct: 1,
                    explanation: "The cerebellum is responsible for balance, coordination, and fine motor control."
                }
            ],
            physiology: [
                {
                    question: "What is the normal resting heart rate for adults?",
                    options: ["40-60 bpm", "60-100 bpm", "100-120 bpm", "120-140 bpm"],
                    correct: 1,
                    explanation: "Normal resting heart rate for adults ranges from 60-100 beats per minute."
                },
                {
                    question: "Which hormone regulates blood sugar levels?",
                    options: ["Cortisol", "Insulin", "Thyroxine", "Adrenaline"],
                    correct: 1,
                    explanation: "Insulin, produced by the pancreas, regulates blood glucose levels."
                }
            ],
            pathology: [
                {
                    question: "What is the most common type of cancer worldwide?",
                    options: ["Breast cancer", "Lung cancer", "Prostate cancer", "Colorectal cancer"],
                    correct: 1,
                    explanation: "Lung cancer is the most common cancer globally and the leading cause of cancer deaths."
                },
                {
                    question: "Which condition is characterized by high blood pressure?",
                    options: ["Hypotension", "Hypertension", "Bradycardia", "Tachycardia"],
                    correct: 1,
                    explanation: "Hypertension is the medical term for high blood pressure."
                }
            ],
            pharmacology: [
                {
                    question: "What is the generic name for Tylenol?",
                    options: ["Ibuprofen", "Acetaminophen", "Aspirin", "Naproxen"],
                    correct: 1,
                    explanation: "Acetaminophen is the generic name for the brand name drug Tylenol."
                },
                {
                    question: "Which class of drugs is used to treat bacterial infections?",
                    options: ["Antivirals", "Antibiotics", "Antifungals", "Antihistamines"],
                    correct: 1,
                    explanation: "Antibiotics are specifically designed to treat bacterial infections."
                }
            ],
            microbiology: [
                {
                    question: "Which microorganism causes tuberculosis?",
                    options: ["Virus", "Bacteria", "Fungus", "Parasite"],
                    correct: 1,
                    explanation: "Tuberculosis is caused by Mycobacterium tuberculosis, a type of bacteria."
                }
            ]
        };

        // Get questions for the selected topic
        let availableQuestions = questionBank[topic] || questionBank.anatomy;
        
        // If we need more questions than available, repeat some
        while (availableQuestions.length < count) {
            availableQuestions = [...availableQuestions, ...questionBank[topic]];
        }

        // Shuffle and select the required number
        const shuffled = this.shuffleArray([...availableQuestions]);
        return shuffled.slice(0, count).map((q, index) => ({
            ...q,
            id: index + 1,
            topic: topic,
            difficulty: difficulty
        }));
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    displayQuestions() {
        const container = document.getElementById('questions-container');
        container.innerHTML = '';

        // Create progress bar
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        progressBar.innerHTML = '<div class="progress-fill" id="progress-fill"></div>';
        container.appendChild(progressBar);

        this.questions.forEach((question, index) => {
            const questionDiv = document.createElement('div');
            questionDiv.className = 'question-item';
            questionDiv.style.display = index === 0 ? 'block' : 'none';
            questionDiv.dataset.questionIndex = index;

            questionDiv.innerHTML = `
                <div class="question-text">
                    ${index + 1}. ${question.question}
                </div>
                <div class="options-container">
                    ${question.options.map((option, optIndex) => `
                        <label class="option-item" data-option="${optIndex}">
                            <input type="radio" name="question_${index}" value="${optIndex}">
                            <span>${String.fromCharCode(65 + optIndex)}. ${option}</span>
                        </label>
                    `).join('')}
                </div>
            `;

            container.appendChild(questionDiv);
        });

        // Add event listeners for option selection
        container.addEventListener('change', (e) => {
            if (e.target.type === 'radio') {
                const questionIndex = parseInt(e.target.name.split('_')[1]);
                const selectedOption = parseInt(e.target.value);
                this.userAnswers[questionIndex] = selectedOption;

                // Update option styling
                const questionDiv = e.target.closest('.question-item');
                questionDiv.querySelectorAll('.option-item').forEach(item => {
                    item.classList.remove('selected');
                });
                e.target.closest('.option-item').classList.add('selected');

                this.updateNavigation();
            }
        });

        this.updateQuestionDisplay();
        this.updateNavigation();
    }

    navigateQuestion(direction) {
        const newIndex = this.currentQuestionIndex + direction;
        
        if (newIndex >= 0 && newIndex < this.questions.length) {
            this.currentQuestionIndex = newIndex;
            this.updateQuestionDisplay();
            this.updateNavigation();
        }
    }

    updateQuestionDisplay() {
        // Hide all questions
        document.querySelectorAll('.question-item').forEach(item => {
            item.style.display = 'none';
        });

        // Show current question
        const currentQuestion = document.querySelector(`[data-question-index="${this.currentQuestionIndex}"]`);
        if (currentQuestion) {
            currentQuestion.style.display = 'block';
        }

        // Update counter
        document.getElementById('question-counter').textContent = 
            `Question ${this.currentQuestionIndex + 1} of ${this.questions.length}`;

        // Update progress bar
        const progress = ((this.currentQuestionIndex + 1) / this.questions.length) * 100;
        document.getElementById('progress-fill').style.width = `${progress}%`;
    }

    updateNavigation() {
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const submitBtn = document.getElementById('submit-btn');

        // Previous button
        prevBtn.disabled = this.currentQuestionIndex === 0;

        // Next/Submit button logic
        const isLastQuestion = this.currentQuestionIndex === this.questions.length - 1;
        const allAnswered = Object.keys(this.userAnswers).length === this.questions.length;

        if (isLastQuestion) {
            nextBtn.style.display = 'none';
            submitBtn.style.display = 'inline-flex';
            submitBtn.disabled = !allAnswered;
        } else {
            nextBtn.style.display = 'inline-flex';
            submitBtn.style.display = 'none';
            nextBtn.disabled = false;
        }
    }

    startTimer() {
        this.startTime = new Date();
        this.timer = setInterval(() => {
            const elapsed = new Date() - this.startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            document.getElementById('timer-display').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }

    stopTimer() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
        this.endTime = new Date();
    }

    submitTest() {
        this.stopTimer();
        this.calculateResults();
        this.displayResults();
        this.showSection('results-section');
    }

    calculateResults() {
        let correctCount = 0;
        const totalQuestions = this.questions.length;

        this.questions.forEach((question, index) => {
            const userAnswer = this.userAnswers[index];
            if (userAnswer === question.correct) {
                correctCount++;
            }
        });

        this.results = {
            correct: correctCount,
            incorrect: totalQuestions - correctCount,
            total: totalQuestions,
            percentage: Math.round((correctCount / totalQuestions) * 100),
            timeTaken: this.endTime - this.startTime
        };
    }

    displayResults() {
        // Update score display
        document.getElementById('score-percentage').textContent = `${this.results.percentage}%`;
        document.getElementById('correct-count').textContent = this.results.correct;
        document.getElementById('incorrect-count').textContent = this.results.incorrect;
        
        // Format time taken
        const minutes = Math.floor(this.results.timeTaken / 60000);
        const seconds = Math.floor((this.results.timeTaken % 60000) / 1000);
        document.getElementById('time-taken').textContent = 
            `${minutes}:${seconds.toString().padStart(2, '0')}`;

        // Performance analysis
        const analysisText = this.getPerformanceAnalysis();
        document.getElementById('analysis-text').innerHTML = analysisText;

        // Question review
        this.displayQuestionReview();

        // Update score circle color based on performance
        const scoreCircle = document.querySelector('.score-circle');
        if (this.results.percentage >= 80) {
            scoreCircle.style.background = 'conic-gradient(#28a745 0deg, #20c997 360deg)';
        } else if (this.results.percentage >= 60) {
            scoreCircle.style.background = 'conic-gradient(#ffc107 0deg, #fd7e14 360deg)';
        } else {
            scoreCircle.style.background = 'conic-gradient(#dc3545 0deg, #e83e8c 360deg)';
        }
    }

    getPerformanceAnalysis() {
        const percentage = this.results.percentage;
        let analysis = '';

        if (percentage >= 90) {
            analysis = `
                <div class="alert alert-success">
                    <strong>Excellent Performance!</strong> You scored ${percentage}%. 
                    You have a strong understanding of the topic. Keep up the great work!
                </div>
            `;
        } else if (percentage >= 80) {
            analysis = `
                <div class="alert alert-info">
                    <strong>Good Performance!</strong> You scored ${percentage}%. 
                    You have a solid grasp of the material with room for minor improvements.
                </div>
            `;
        } else if (percentage >= 60) {
            analysis = `
                <div class="alert alert-warning">
                    <strong>Fair Performance.</strong> You scored ${percentage}%. 
                    Consider reviewing the material and practicing more questions.
                </div>
            `;
        } else {
            analysis = `
                <div class="alert alert-danger">
                    <strong>Needs Improvement.</strong> You scored ${percentage}%. 
                    We recommend studying the topic more thoroughly and retaking the test.
                </div>
            `;
        }

        return analysis;
    }

    displayQuestionReview() {
        const container = document.getElementById('review-container');
        container.innerHTML = '';

        this.questions.forEach((question, index) => {
            const userAnswer = this.userAnswers[index];
            const isCorrect = userAnswer === question.correct;
            
            const reviewItem = document.createElement('div');
            reviewItem.className = `review-item ${isCorrect ? 'correct' : ''}`;
            
            reviewItem.innerHTML = `
                <div class="review-question">
                    ${index + 1}. ${question.question}
                </div>
                <div class="review-answer">
                    <span>Your Answer: ${userAnswer !== undefined ? question.options[userAnswer] : 'Not answered'}</span>
                    <span class="${isCorrect ? 'text-success' : 'text-danger'}">
                        ${isCorrect ? '✓ Correct' : '✗ Incorrect'}
                    </span>
                </div>
                ${!isCorrect ? `
                    <div class="correct-answer">
                        <strong>Correct Answer:</strong> ${question.options[question.correct]}
                    </div>
                    <div class="explanation">
                        <strong>Explanation:</strong> ${question.explanation}
                    </div>
                ` : ''}
            `;

            container.appendChild(reviewItem);
        });
    }

    restart() {
        this.questions = [];
        this.currentQuestionIndex = 0;
        this.userAnswers = {};
        this.startTime = null;
        this.endTime = null;
        
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }

        // Reset form
        document.getElementById('question-form').reset();
        
        this.showSection('generation-section');
    }

    downloadResults() {
        const results = {
            timestamp: new Date().toISOString(),
            topic: this.questions[0]?.topic || 'Unknown',
            difficulty: this.questions[0]?.difficulty || 'Unknown',
            score: this.results,
            questions: this.questions.map((q, index) => ({
                question: q.question,
                userAnswer: this.userAnswers[index] !== undefined ? q.options[this.userAnswers[index]] : 'Not answered',
                correctAnswer: q.options[q.correct],
                isCorrect: this.userAnswers[index] === q.correct
            }))
        };

        const dataStr = JSON.stringify(results, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `medical_test_results_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MedicalQuestionApp();
});

// Add some CSS for alerts
const alertStyles = `
    .alert {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .alert-success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
    .alert-info { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
    .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
    .alert-danger { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    .correct-answer, .explanation { 
        margin-top: 10px; 
        padding: 8px; 
        background: rgba(0,0,0,0.05); 
        border-radius: 4px; 
        font-size: 0.9rem; 
    }
`;

const style = document.createElement('style');
style.textContent = alertStyles;
document.head.appendChild(style);
