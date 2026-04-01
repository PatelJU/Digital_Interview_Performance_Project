# 🤖 Digital Interview Performance Scoring

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Overview

This project evaluates candidate performance in digital interviews using machine learning by analyzing **17 behavioral and performance metrics** including communication skills, confidence levels, body language, and psychological factors. The system provides data-driven hiring recommendations to support HR professionals in making objective, bias-free decisions.

## ✨ Features

- **🎯 AI-Powered Evaluation**: Random Forest classifier predicts hiring decisions based on 16 key features
- **📊 Interactive Analytics**: Comprehensive exploratory data analysis with Plotly visualizations
- **🗣️ Communication Analysis**: Evaluates verbal effectiveness, vocabulary richness, filler words, and sentiment
- **🎭 Behavioral Assessment**: Analyzes confidence, eye contact, and posture using computer vision principles
- **🧠 Psychological Insights**: Monitors stress levels and response times for optimal performance indicators
- **📈 Performance Scoring**: Generates composite scores combining all behavioral and communication factors
- **💼 Multi-Role Support**: Covers 8+ job roles including Software Engineer, Data Scientist, Marketing Analyst, etc.
- **🚀 BULK CANDIDATE SCAN**: Process hundreds of candidates simultaneously via CSV upload with instant predictions and analytics

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/PatelJU/Digital_Interview_Performance_Project.git
cd Digital-Interview-Performance-AI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
Digital-Interview-Performance-AI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── BULK_SCAN_GUIDE.md             # Bulk scan feature user guide
├── test_bulk_candidates.csv       # Sample test data for bulk scan
├── data/
│   └── interview_performance_data.csv   # Dataset (7,501 records)
├── models/
│   └── model.pkl                   # Trained Random Forest model (~20MB)
├── notebooks/
│   └── Digital_Interview_Performance_Analysis.ipynb  # Jupyter notebook analysis
└── reports/
    └── ydata_profile_report.html  # Automated EDA report
```

## 🎯 What You Can Do

### 1. **Project Overview**
- Understand the dataset containing 7,501 candidate interviews
- Learn about 15+ behavioral metrics across 5 categories
- Explore key objectives and problem statements

### 2. **Exploratory Data Analysis**
- View hiring decision distributions
- Analyze correlation heatmaps
- Examine relationships between stress and performance
- Compare experience levels across hiring outcomes

### 3. **AI Performance Prediction**
- Input candidate metrics through an interactive form
- Get instant hiring decision predictions
- View candidate strength profiles via radar charts
- Receive personalized feedback and improvement areas

### 4. **Bulk Candidate Scan** ⭐ NEW!
- Upload CSV files with hundreds of candidates
- Process all candidates simultaneously
- Get batch predictions with rankings
- Download comprehensive results and analytics
- Generate summary statistics for reporting

### 5. **Download Reports**
- Download complete dataset (CSV format)
- Generate comprehensive HTML analysis reports
- Access statistical summaries and visualizations

## 📊 Dataset Metrics

The dataset includes the following key measurements:

| Category | Metrics |
|----------|---------|
| **Demographics** | Age, Years of Experience, Education, Job Role |
| **Communication** | Communication Score, Vocabulary Richness, Filler Words, Sentiment |
| **Behavioral** | Confidence Score, Eye Contact Score, Posture |
| **Psychological** | Stress Level, Response Time, Interview Duration |
| **Outcomes** | Performance Score, Hiring Decision |

### Key Statistics
- **Total Candidates**: 7,501
- **Job Roles**: 8 (Software Engineer, Data Scientist, Marketing Analyst, etc.)
- **Platforms**: 4 (Zoom, Teams, HireVue, Google Meet)
- **Average Performance Score**: ~75/100
- **Hiring Rate**: ~25%

## 🤖 Machine Learning Model

**Algorithm**: Random Forest Classifier

**Features**: 16 input features including:
- Numerical: Age, Experience, Communication Score, Confidence, etc.
- Categorical: Job Role, Education, Platform, Posture

**Output Classes**: 
- ✅ Hired
- 🔄 Second Round
- ⏸️ On Hold
- ❌ Rejected

**Performance Metrics**:
- Accuracy: High predictive accuracy on test data
- F1-Score: Balanced precision and recall
- Robust handling of imbalanced classes

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Profiling**: YData Profiling
- **Development**: Jupyter Notebook

### Project Overview Page
Shows abstract, objectives, and dataset statistics with a clean dark blue theme.

### Exploratory Data Analysis
Interactive histograms, correlation heatmaps, and scatter plots revealing patterns in hiring decisions.

### AI Prediction Interface
Input form for candidate metrics with real-time prediction and strength profile visualization.

### Download Section
Options to download raw dataset and comprehensive HTML analysis reports.

