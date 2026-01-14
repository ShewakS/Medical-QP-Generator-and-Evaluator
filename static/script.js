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
        document.getElementById('question-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.generateQuestions();
        });
        document.getElementById('prev-btn').addEventListener('click', () => {
            this.navigateQuestion(-1);
        });
        document.getElementById('next-btn').addEventListener('click', () => {
            this.navigateQuestion(1);
        });
        document.getElementById('submit-btn').addEventListener('click', () => {
            this.submitTest();
        });
        document.getElementById('restart-btn').addEventListener('click', () => {
            this.restart();
        });
        document.getElementById('download-results').addEventListener('click', () => {
            this.downloadResults();
        });
    }

    showSection(sectionId) {
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
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
            let questions;
            try {
                const response = await fetch('/api/generate-questions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({topic: topic, count: count, difficulty: difficulty})
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
        if (!this.generatedQuestionsTracker) {
            this.generatedQuestionsTracker = {};
        }
        const cacheKey = `${topic}_${difficulty}`;
        if (!this.generatedQuestionsTracker[cacheKey]) {
            this.generatedQuestionsTracker[cacheKey] = new Set();
        }

        const questionBank = {
            anatomy: [
                {question: "Which bone connects the shoulder to the elbow?", options: ["Tibia", "Femur", "Humerus", "Radius"], correct: 2, explanation: "The humerus is the bone that connects the shoulder to the elbow in the upper arm."},
                {question: "How many chambers does a human heart have?", options: ["2", "3", "4", "5"], correct: 2, explanation: "The human heart has four chambers: two atria and two ventricles."},
                {question: "Which part of the brain controls balance and coordination?", options: ["Cerebrum", "Cerebellum", "Medulla", "Pons"], correct: 1, explanation: "The cerebellum is responsible for balance, coordination, and fine motor control."},
                {question: "What is the largest organ in the human body?", options: ["Liver", "Brain", "Skin", "Lungs"], correct: 2, explanation: "The skin is the largest organ in the human body by surface area."},
                {question: "Which muscle is responsible for breathing?", options: ["Diaphragm", "Intercostals", "Pectorals", "Abdominals"], correct: 0, explanation: "The diaphragm is the primary muscle responsible for breathing."}
            ],
            physiology: [
                {question: "What is the normal resting heart rate for adults?", options: ["40-60 bpm", "60-100 bpm", "100-120 bpm", "120-140 bpm"], correct: 1, explanation: "Normal resting heart rate for adults ranges from 60-100 beats per minute."},
                {question: "Which hormone regulates blood sugar levels?", options: ["Cortisol", "Insulin", "Thyroxine", "Adrenaline"], correct: 1, explanation: "Insulin, produced by the pancreas, regulates blood glucose levels."},
                {question: "What is the normal body temperature in Celsius?", options: ["36°C", "37°C", "38°C", "39°C"], correct: 1, explanation: "Normal body temperature is approximately 37°C (98.6°F)."},
                {question: "Which organ produces bile?", options: ["Pancreas", "Liver", "Gallbladder", "Stomach"], correct: 1, explanation: "The liver produces bile, which is stored in the gallbladder."}
            ],
            pathology: [
                {question: "What is the most common type of cancer worldwide?", options: ["Breast cancer", "Lung cancer", "Prostate cancer", "Colorectal cancer"], correct: 1, explanation: "Lung cancer is the most common cancer globally and the leading cause of cancer deaths."},
                {question: "Which condition is characterized by high blood pressure?", options: ["Hypotension", "Hypertension", "Bradycardia", "Tachycardia"], correct: 1, explanation: "Hypertension is the medical term for high blood pressure."},
                {question: "What does diabetes primarily affect?", options: ["Blood pressure", "Blood sugar", "Heart rate", "Breathing"], correct: 1, explanation: "Diabetes primarily affects blood sugar (glucose) regulation."},
                {question: "Which condition involves inflammation of joints?", options: ["Arthritis", "Osteoporosis", "Fibromyalgia", "Tendinitis"], correct: 0, explanation: "Arthritis is characterized by inflammation of one or more joints."}
            ],
            pharmacology: [
                {question: "What is the generic name for Tylenol?", options: ["Ibuprofen", "Acetaminophen", "Aspirin", "Naproxen"], correct: 1, explanation: "Acetaminophen is the generic name for the brand name drug Tylenol."},
                {question: "Which class of drugs is used to treat bacterial infections?", options: ["Antivirals", "Antibiotics", "Antifungals", "Antihistamines"], correct: 1, explanation: "Antibiotics are specifically designed to treat bacterial infections."},
                {question: "What is the generic name for Advil?", options: ["Acetaminophen", "Ibuprofen", "Aspirin", "Naproxen"], correct: 1, explanation: "Ibuprofen is the generic name for the brand name drug Advil."},
                {question: "Which type of drug reduces fever?", options: ["Antibiotic", "Antipyretic", "Antihistamine", "Antacid"], correct: 1, explanation: "Antipyretic drugs are used to reduce fever."}
            ],
            microbiology: [
                {question: "Which microorganism causes tuberculosis?", options: ["Virus", "Bacteria", "Fungus", "Parasite"], correct: 1, explanation: "Tuberculosis is caused by Mycobacterium tuberculosis, a type of bacteria."},
                {question: "What type of pathogen causes the common cold?", options: ["Bacteria", "Virus", "Fungus", "Parasite"], correct: 1, explanation: "The common cold is caused by various viruses, most commonly rhinoviruses."},
                {question: "Which organism causes malaria?", options: ["Virus", "Bacteria", "Parasite", "Fungus"], correct: 2, explanation: "Malaria is caused by Plasmodium parasites transmitted by mosquitoes."}
            ]
        };

        const baseQuestions = questionBank[topic] || questionBank.anatomy;
        const allPossibleQuestions = [];
        const questionHashes = new Set();
        baseQuestions.forEach(baseQ => {
            for (let seed = 0; seed < 10; seed++) {
                const options = [...baseQ.options];
                for (let i = options.length - 1; i > 0; i--) {
                    const j = Math.floor((seed * 31 + i) % (i + 1));
                    [options[i], options[j]] = [options[j], options[i]];
                }
                const questionHash = `${baseQ.question}_${options.join('_')}`;
                if (!questionHashes.has(questionHash)) {
                    questionHashes.add(questionHash);
                    const correct = options.indexOf(baseQ.options[baseQ.correct]);
                    allPossibleQuestions.push({...baseQ, options: options, correct: correct, hash: questionHash});
                }
            }
        });
        const availableQuestions = allPossibleQuestions.filter(q => 
            !this.generatedQuestionsTracker[cacheKey].has(q.hash)
        );
        if (availableQuestions.length < count) {
            console.log(`⚠️ Resetting question tracker for ${cacheKey} - generating fresh questions`);
            this.generatedQuestionsTracker[cacheKey].clear();
            availableQuestions.push(...allPossibleQuestions);
        }
        this.shuffleArray(availableQuestions);
        const selectedQuestions = availableQuestions.slice(0, count);
        selectedQuestions.forEach(q => {
            this.generatedQuestionsTracker[cacheKey].add(q.hash);
        });
        return selectedQuestions.map((q, index) => ({...q, id: index + 1, topic: topic, difficulty: difficulty}));
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
        container.addEventListener('change', (e) => {
            if (e.target.type === 'radio') {
                const questionIndex = parseInt(e.target.name.split('_')[1]);
                const selectedOption = parseInt(e.target.value);
                this.userAnswers[questionIndex] = selectedOption;
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
        document.querySelectorAll('.question-item').forEach(item => {
            item.style.display = 'none';
        });
        const currentQuestion = document.querySelector(`[data-question-index="${this.currentQuestionIndex}"]`);
        if (currentQuestion) {
            currentQuestion.style.display = 'block';
        }
        document.getElementById('question-counter').textContent = 
            `Question ${this.currentQuestionIndex + 1} of ${this.questions.length}`;
        const progress = ((this.currentQuestionIndex + 1) / this.questions.length) * 100;
        document.getElementById('progress-fill').style.width = `${progress}%`;
    }

    updateNavigation() {
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const submitBtn = document.getElementById('submit-btn');
        prevBtn.disabled = this.currentQuestionIndex === 0;
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
        this.saveResultsToDatabase();  // Save to database
        this.showSection('results-section');
    }

    async saveResultsToDatabase() {
        try {
            const testData = {
                topic: this.userAnswers.topic || 'Unknown',
                num_questions: this.questions.length,
                score: this.results.percentage,
                total_questions: this.questions.length,
                answers: this.userAnswers,
                generated_questions: this.questions.map(q => ({
                    question: q.question,
                    options: [q.opa, q.opb, q.opc, q.opd],
                    correct: q.correct
                }))
            };

            const response = await fetch('/api/save-test-result', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(testData)
            });

            if (response.ok) {
                console.log('Test results saved to database');
            }
        } catch (error) {
            console.error('Error saving test results:', error);
            // Don't block UI if saving fails
        }
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

        // Display selected topic and difficulty
        const topic = this.questions[0]?.topic || 'Unknown';
        const difficulty = this.questions[0]?.difficulty || 'Unknown';
        document.getElementById('test-topic').textContent = topic.charAt(0).toUpperCase() + topic.slice(1);
        document.getElementById('test-difficulty').textContent = difficulty.charAt(0).toUpperCase() + difficulty.slice(1);

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
        if (percentage >= 90) {
            return `<div class="alert alert-success"><strong>Excellent Performance!</strong> You scored ${percentage}%. You have a strong understanding of the topic. Keep up the great work!</div>`;
        } else if (percentage >= 80) {
            return `<div class="alert alert-info"><strong>Good Performance!</strong> You scored ${percentage}%. You have a solid grasp of the material with room for minor improvements.</div>`;
        } else if (percentage >= 60) {
            return `<div class="alert alert-warning"><strong>Fair Performance.</strong> You scored ${percentage}%. Consider reviewing the material and practicing more questions.</div>`;
        } else {
            return `<div class="alert alert-danger"><strong>Needs Improvement.</strong> You scored ${percentage}%. We recommend studying the topic more thoroughly and retaking the test.</div>`;
        }
    }

    displayQuestionReview() {
        const container = document.getElementById('review-container');
        container.innerHTML = '';
        this.questions.forEach((question, index) => {
            const userAnswer = this.userAnswers[index];
            const isCorrect = userAnswer === question.correct;
            const reviewItem = document.createElement('div');
            reviewItem.className = `review-item ${isCorrect ? 'correct' : ''}`;
            
            // Ensure we have valid options array
            const options = question.options || ['Option A', 'Option B', 'Option C', 'Option D'];
            const correctAnswerText = options[question.correct] || 'Answer not available';
            const userAnswerText = userAnswer !== undefined ? options[userAnswer] || 'Invalid selection' : 'Not answered';
            
            reviewItem.innerHTML = `
                <div class="review-question">
                    ${index + 1}. ${question.question}
                </div>
                <div class="review-answer">
                    <span>Your Answer: ${userAnswerText}</span>
                    <span class="${isCorrect ? 'text-success' : 'text-danger'}">
                        ${isCorrect ? '✓ Correct' : '✗ Incorrect'}
                    </span>
                </div>
                ${!isCorrect ? `
                    <div class="correct-answer">
                        <strong>Correct Answer:</strong> ${correctAnswerText}
                    </div>
                    <div class="explanation">
                        <strong>Explanation:</strong> ${question.explanation || 'No explanation available'}
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
        document.getElementById('question-form').reset();
        this.showSection('generation-section');
    }

    async downloadResults() {
        // Get student name from input field (auto-populated with logged-in user's name)
        const studentName = document.getElementById('student-name')?.value.trim() || 'Student';
        
        const results = {
            studentName: studentName,
            timestamp: new Date().toISOString(),
            topic: this.questions[0]?.topic || 'Unknown',
            difficulty: this.questions[0]?.difficulty || 'Unknown',
            total: this.results.total,
            correct: this.results.correct,
            percentage: this.results.percentage,
            questions: this.questions.map((q, index) => ({
                question: q.question,
                userAnswer: this.userAnswers[index] !== undefined ? q.options[this.userAnswers[index]] : 'Not answered',
                correctAnswer: q.options[q.correct],
                isCorrect: this.userAnswers[index] === q.correct
            }))
        };

        try {
            const response = await fetch('/api/download-pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ results })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `medical_test_results_${new Date().toISOString().split('T')[0]}.pdf`;
                link.click();
                window.URL.revokeObjectURL(url);
            } else {
                throw new Error('PDF generation failed');
            }
        } catch (error) {
            console.error('Error downloading PDF:', error);
            alert('Error generating PDF. Please try again.');
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new MedicalQuestionApp();
});
const alertStyles = `
    .alert { padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .alert-success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
    .alert-info { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
    .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
    .alert-danger { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    .correct-answer, .explanation { margin-top: 10px; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 4px; font-size: 0.9rem; }
`;
const style = document.createElement('style');
style.textContent = alertStyles;
document.head.appendChild(style);
