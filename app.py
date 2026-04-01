import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as px_go
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Digital Interview Performance AI", page_icon="🤖", layout="wide")

# --- CUSTOM CSS (Dark Blue Theme) ---
st.markdown(
    """
    <style>
    .main {
        background-color: #0a192f !important;
    }
    .stMetric {
        background: linear-gradient(135deg, #112d4e 0%, #0a192f 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: #e6f1ff;
    }
    .sidebar .sidebar-content {
        background-color: #020c1b !important;
    }
    .sidebar .sidebar-content .stSidebarContent {
        background-color: #020c1b !important;
    }
    .explanation-box {
        background: linear-gradient(135deg, #112d4e 0%, #0f3460 100%);
        border-left: 4px solid #00d9ff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #e6f1ff;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a3a52 0%, #2d5a3f 100%);
        border-left: 4px solid #ffd700;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #e6f1ff;
    }
    .metric-card {
        background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,217,255,0.3);
    }
    .tip-box {
        background: linear-gradient(135deg, #112d4e 0%, #0f3460 100%);
        border-left: 4px solid #00ff88;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #e6f1ff;
    }
    .tip-box strong {
        color: #00ff88;
    }
    .tip-box p {
        color: #e6f1ff;
    }
    /* Override Streamlit default backgrounds */
    .block-container {
        background-color: #0a192f;
    }
    .stApp {
        background-color: #0a192f;
    }
    /* Text colors for dark theme */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #e6f1ff !important;
    }
    /* Fix for headings */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #00d9ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    # Try multiple paths for the dataset
    paths = ['data/interview_performance_data.csv', 'interview_performance_data.csv', '../data/interview_performance_data.csv']
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

@st.cache_resource
def load_model():
    # Try multiple paths for the model
    paths = ['models/model.pkl', 'model.pkl', '../models/model.pkl']
    for p in paths:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    return None

df = load_data()
model = load_model()

# --- GENERATE HTML REPORT FUNCTION ---
def generate_html_report(df):
    """Generate a comprehensive HTML report with all visualizations and analysis"""
    
    # Data cleaning
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Calculate statistics
    total_candidates = len(df)
    avg_performance = df['performance_score'].mean()
    hiring_rate = (df['hiring_decision'] == 'Hired').mean() * 100
    avg_communication = df['communication_score'].mean()
    avg_confidence = df['confidence_score'].mean()
    avg_experience = df['years_of_experience'].mean()
    
    # Hiring decision counts
    hiring_counts = df['hiring_decision'].value_counts()
    
    # Correlation matrix
    corr = df_clean.select_dtypes(include=[np.number]).corr()
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Interview Performance Scoring - Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f7f6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card h3 {{
            margin: 0;
            font-size: 2em;
        }}
        .metric-card p {{
            margin: 5px 0 0 0;
            opacity: 0.9;
        }}
        .info-box {{
            background-color: #e8f4fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .insight-box {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .plot-container {{
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            color: #666;
        }}
        @media print {{
            body {{
                background: white;
            }}
            .section {{
                box-shadow: none;
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Digital Interview Performance Scoring</h1>
        <p>Comprehensive Analysis Report | AI-Powered HR Analytics</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>📋 Executive Summary</h2>
        <div class="info-box">
            <p>This report presents a comprehensive analysis of <strong>{total_candidates:,} candidate interviews</strong> using machine learning and data analytics. The system evaluates candidates based on <strong>17 behavioral and performance metrics</strong> including communication skills, behavioral indicators, psychological factors, and interview outcomes.</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>{total_candidates:,}</h3>
                <p>Total Candidates</p>
            </div>
            <div class="metric-card">
                <h3>{avg_performance:.1f}</h3>
                <p>Avg Performance Score</p>
            </div>
            <div class="metric-card">
                <h3>{hiring_rate:.1f}%</h3>
                <p>Hiring Rate</p>
            </div>
            <div class="metric-card">
                <h3>{avg_experience:.1f}</h3>
                <p>Avg Experience (Years)</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📊 Dataset Overview</h2>
        <p>The dataset contains the following key metrics for each candidate:</p>
        <table>
            <tr>
                <th>Category</th>
                <th>Metrics</th>
            </tr>
            <tr>
                <td><strong>Demographics</strong></td>
                <td>Age, Years of Experience</td>
            </tr>
            <tr>
                <td><strong>Communication</strong></td>
                <td>Communication Score, Vocabulary Richness, Filler Words, Sentiment</td>
            </tr>
            <tr>
                <td><strong>Behavioral</strong></td>
                <td>Confidence, Eye Contact, Posture</td>
            </tr>
            <tr>
                <td><strong>Psychological</strong></td>
                <td>Stress Level, Response Time, Interview Duration</td>
            </tr>
            <tr>
                <td><strong>Outcomes</strong></td>
                <td>Performance Score, Hiring Decision</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>📈 Key Findings</h2>
        
        <h3>Hiring Decision Distribution</h3>
        <div class="info-box">
            <p><strong>Hired:</strong> {hiring_counts.get('Hired', 0)} candidates ({hiring_counts.get('Hired', 0)/total_candidates*100:.1f}%)<br>
            <strong>Rejected:</strong> {hiring_counts.get('Rejected', 0)} candidates ({hiring_counts.get('Rejected', 0)/total_candidates*100:.1f}%)<br>
            <strong>Second Round:</strong> {hiring_counts.get('Second Round', 0)} candidates ({hiring_counts.get('Second Round', 0)/total_candidates*100:.1f}%)<br>
            <strong>On Hold:</strong> {hiring_counts.get('On Hold', 0)} candidates ({hiring_counts.get('On Hold', 0)/total_candidates*100:.1f}%)</p>
        </div>
        
        <h3>Performance Statistics</h3>
        <ul>
            <li><strong>Average Communication Score:</strong> {avg_communication:.1f}/100</li>
            <li><strong>Average Confidence Score:</strong> {avg_confidence:.1f}/100</li>
            <li><strong>Average Performance Score:</strong> {avg_performance:.2f}/100</li>
            <li><strong>Standard Deviation:</strong> {df['performance_score'].std():.2f}</li>
        </ul>
    </div>

    <div class="section">
        <h2>🔍 Correlation Analysis</h2>
        <div class="insight-box">
            <p>The correlation matrix reveals relationships between different interview metrics. Strong positive correlations with performance_score indicate key success factors, while negative correlations highlight areas that may hinder candidate success.</p>
        </div>
        <p><em>Note: Interactive plots are best viewed in the web version. This report shows static representations.</em></p>
    </div>

    <div class="section">
        <h2>💡 Recommendations for HR Professionals</h2>
        <div class="insight-box">
            <ul>
                <li><strong>Focus on Communication:</strong> Communication scores show strong correlation with overall performance</li>
                <li><strong>Manage Stress Levels:</strong> Excessive stress negatively impacts candidate performance</li>
                <li><strong>Consider Experience:</strong> Years of experience correlate with better hiring outcomes</li>
                <li><strong>Use Data-Driven Approach:</strong> AI-powered evaluation reduces bias and improves hiring quality</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>🤖 Machine Learning Model</h2>
        <p><strong>Algorithm:</strong> Random Forest Classifier</p>
        <p><strong>Features:</strong> 16 input features across 5 categories</p>
        <p><strong>Output Classes:</strong> Hired, Rejected, Second Round, On Hold</p>
        <div class="info-box">
            <p>The model was trained on historical interview data to predict hiring decisions based on behavioral and performance metrics. It provides unbiased, consistent evaluation across all candidates.</p>
        </div>
    </div>

    <div class="footer">
        <p><strong>Digital Interview Performance Scoring Project</strong></p>
        <p>Powered by Machine Learning | Computer Vision | NLP</p>
        <p>Report generated for educational and research purposes</p>
    </div>
</body>
</html>
"""
    return html_content

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Exploratory Data Analysis", "AI Performance Prediction", "Download Reports"])

# --- 1. PROJECT OVERVIEW ---
if page == "Project Overview":
    st.title("🤖 Digital Interview Performance Scoring")
    st.markdown("---")
    
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Abstract")
        st.write("""
        This project uses machine learning to evaluate candidate performance in digital interviews by analyzing facial expressions, 
        tone, body language, and speech quality. Computer vision models extract visual cues such as eye contact, confidence level, 
        and engagement. NLP techniques assess verbal clarity, sentiment, and communication effectiveness. 
        The system produces a performance score that helps HR teams make data-driven hiring decisions.
        """)
        
        st.header("Key Objectives")
        st.markdown("""
        - **Analyze Behavioral Traits**: Understand how communication, confidence, and stress impact hiring.
        - **Data-Driven Recruitment**: Reduce manual bias and support scalable remote hiring.
        - **Predictive Modeling**: Build a robust AI model to categorize candidates accurately.
        """)
        
        st.info("**💡 What Makes This System Unique?**\n\n" +
                "Our AI-powered platform analyzes **15+ behavioral metrics** simultaneously, providing objective, bias-free evaluation " +
                "of candidates in digital interviews. By leveraging computer vision and natural language processing, we transform " +
                "subjective interview assessments into quantifiable, actionable insights.")
    
    with col2:
        st.info("**Keywords:** Data Science, ML, HR Analytics, Computer Vision, NLP")
        if df is not None:
            st.metric("Total Candidates", len(df))
            st.metric("Avg Performance Score", f"{df['performance_score'].mean():.2f}")
            st.metric("Hiring Rate", f"{(df['hiring_decision'] == 'Hired').mean()*100:.1f}%")
    
    # Detailed Metrics Section
    st.markdown("---")
    st.subheader("📊 Dataset Overview & Key Insights")
    st.markdown("""
    <div class='explanation-box'>
        <strong>📋 About the Dataset:</strong><br>
        This dataset contains <strong>7,501 candidate interviews</strong> across multiple job roles including Software Engineer, 
        Data Scientist, Marketing Analyst, Sales Executive, HR Manager, Business Analyst, UX Designer, and Product Manager. 
        Each candidate is evaluated on <strong>15 behavioral and performance metrics</strong> captured during video interviews 
        conducted on platforms like Zoom, Microsoft Teams, HireVue, and Google Meet.
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_age = df['age'].mean()
        st.metric(label="👤 Average Age", value=f"{avg_age:.1f} years",
                 delta=f"Range: {df['age'].min()}-{df['age'].max()} years")
    
    with col2:
        avg_exp = df['years_of_experience'].mean()
        st.metric(label="💼 Avg Experience", value=f"{avg_exp:.1f} years",
                 delta=f"Range: {df['years_of_experience'].min()}-{df['years_of_experience'].max()} years")
    
    with col3:
        avg_comm = df['communication_score'].mean()
        st.metric(label="🗣️ Avg Communication", value=f"{avg_comm:.1f}/100",
                 delta="Communication effectiveness score")
    
    with col4:
        avg_conf = df['confidence_score'].mean()
        st.metric(label="😊 Avg Confidence", value=f"{avg_conf:.1f}/100",
                 delta="Confidence level indicator")
    
    st.markdown("---")
    
    # Feature Categories Explanation
    st.subheader("🔍 What Do We Measure?")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📢 Communication Metrics",
        "🎭 Behavioral Indicators",
        "🧠 Psychological Factors",
        "📈 Performance Outcomes"
    ])
    
    with tab1:
        st.markdown("""
        ### 🗣️ Communication Metrics
        
        These metrics analyze how effectively candidates communicate:
        
        - **Communication Score (0-150)**: Overall effectiveness of verbal communication, including clarity, articulation, and coherence
        - **Vocabulary Richness (0-1)**: Diversity and sophistication of word usage (higher = more varied vocabulary)
        - **Filler Words per Min (0-20)**: Frequency of hesitation words like 'um', 'uh', 'like' (lower is better)
        - **Sentiment Score (-1 to +1)**: Emotional tone detected in speech (+1 = very positive, -1 = very negative)
        
        <div class='insight-box'>
            <strong>💡 Why It Matters:</strong> Strong communication skills are critical for most roles. Candidates who speak clearly, 
            use rich vocabulary, and minimize filler words typically perform better in professional settings.
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        ### 🎭 Behavioral Indicators
        
        Computer vision analyzes non-verbal cues:
        
        - **Confidence Score (0-100)**: AI-assessed confidence level based on voice tone, posture, and speech patterns
        - **Eye Contact Score (0-100)**: Percentage of time maintaining appropriate eye contact with camera
        - **Posture**: Body language classification (Confident, Neutral, or Slouching)
        
        <div class='insight-box'>
            <strong>💡 Why It Matters:</strong> Non-verbal communication accounts for ~55% of overall communication effectiveness. 
            Strong eye contact and confident posture signal engagement and self-assurance.
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        ### 🧠 Psychological Factors
        
        Mental and emotional state indicators:
        
        - **Stress Level (0-100)**: Physiological stress indicators from voice tremors, facial tension, and body movements
        - **Avg Response Time (seconds)**: Time taken to respond to questions (optimal range: 3-6 seconds)
        - **Interview Duration (minutes)**: Total length of the interview session
        
        <div class='insight-box'>
            <strong>💡 Why It Matters:</strong> Moderate stress can enhance performance, but excessive stress impairs cognitive function. 
            Response time indicates thoughtfulness vs. overthinking.
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        ### 📈 Performance Outcomes
        
        Final assessment metrics:
        
        - **Performance Score (0-100)**: Composite AI-generated score combining all behavioral and communication factors
        - **Hiring Decision**: Final recommendation (Hired, Rejected, On Hold, Second Round)
        
        <div class='explanation-box'>
            <strong>📊 Scoring Methodology:</strong> The performance score is calculated using a weighted combination of all measured 
            attributes, with different weights potentially applied based on job role requirements. This score directly influences 
            the final hiring decision recommendation.
        </div>
        """, unsafe_allow_html=True)

# --- 2. EXPLORATORY DATA ANALYSIS ---
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    st.markdown("""
    <div class='explanation-box'>
        <strong>🔍 Purpose of EDA:</strong> Exploratory Data Analysis helps us understand patterns, relationships, and anomalies in 
        interview data. By visualizing distributions, correlations, and trends, we gain insights into what factors contribute to 
        candidate success and hiring decisions.
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Data Cleaning (same as notebook)
        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "💡 Deep Insights"])
        
        with tab1:
            st.subheader("Target Variable Distribution")
            st.markdown("""
            <div class='explanation-box'>
                <strong>📊 What This Shows:</strong> This histogram displays how hiring decisions are distributed across all candidates. 
                Understanding this distribution helps identify potential biases in the hiring process and reveals whether our dataset 
                is balanced across different outcomes.
            </div>
            """, unsafe_allow_html=True)
            
            fig_hiring = px.histogram(df_clean, x='hiring_decision', 
                                     title="Distribution of Hiring Decisions Across All Candidates", 
                                     color='hiring_decision', 
                                     color_discrete_sequence=px.colors.qualitative.Pastel,
                                     labels={'hiring_decision': 'Hiring Decision', 'count': 'Number of Candidates'})
            fig_hiring.update_layout(showlegend=False)
            st.plotly_chart(fig_hiring, use_container_width=True)
            
            # Add statistical insight
            hiring_counts = df_clean['hiring_decision'].value_counts()
            hire_rate = (df_clean['hiring_decision'] == 'Hired').sum() / len(df_clean) * 100
            st.markdown(f"""
            <div class='insight-box'>
                <strong>💡 Key Insight:</strong> Out of {len(df_clean):,} total candidates:<br>
                • <strong>{hiring_counts.get('Hired', 0)}</strong> candidates were hired ({hire_rate:.1f}%)<br>
                • <strong>{hiring_counts.get('Rejected', 0)}</strong> candidates were rejected<br>
                • <strong>{hiring_counts.get('Second Round', 0)}</strong> advanced to second round<br>
                • <strong>{hiring_counts.get('On Hold', 0)}</strong> decisions are on hold<br>
                <br>
                This distribution shows the competitiveness of the selection process and helps validate if our AI model 
                is trained on representative data.
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Performance Score Distribution")
            st.markdown("""
            <div class='explanation-box'>
                <strong>📊 What This Shows:</strong> This distribution reveals how performance scores are spread across candidates. 
                The shape of this distribution tells us whether the scoring system differentiates well between candidates or if 
                scores cluster around certain values.
            </div>
            """, unsafe_allow_html=True)
            
            fig_perf = px.histogram(df_clean, x='performance_score', 
                                   title="Distribution of Candidate Performance Scores", 
                                   color_discrete_sequence=['#3498db'],
                                   nbins=30,
                                   labels={'performance_score': 'Performance Score (0-100)', 'count': 'Number of Candidates'})
            fig_perf.add_vline(x=df_clean['performance_score'].mean(), line_dash="dash", line_color="red", 
                              annotation_text=f"Mean: {df_clean['performance_score'].mean():.2f}")
            st.plotly_chart(fig_perf, use_container_width=True)
            
            st.markdown(f"""
            <div class='insight-box'>
                <strong>💡 Key Insights:</strong><br>
                • <strong>Average Score:</strong> {df_clean['performance_score'].mean():.2f} out of 100<br>
                • <strong>Standard Deviation:</strong> {df_clean['performance_score'].std():.2f} (measures score variability)<br>
                • <strong>Minimum Score:</strong> {df_clean['performance_score'].min():.2f}<br>
                • <strong>Maximum Score:</strong> {df_clean['performance_score'].max():.2f}<br>
                <br>
                A normal-like distribution suggests the scoring system effectively differentiates candidates across a range of performance levels.
            </div>
            """, unsafe_allow_html=True)
            
        with tab2:
            st.subheader("Feature Correlation Heatmap")
            st.markdown("""
            <div class='explanation-box'>
                <strong>🔗 What This Shows:</strong> A correlation heatmap displays relationships between pairs of variables. 
                Values close to +1 indicate strong positive relationships (both increase together), while values close to -1 
                indicate strong negative relationships (one increases as the other decreases). Values near 0 suggest no linear relationship.
            </div>
            """, unsafe_allow_html=True)
            
            corr = df_clean.select_dtypes(include=[np.number]).corr()
            fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto", 
                                color_continuous_scale='RdBu_r',
                                title="Correlation Matrix: How Interview Metrics Relate to Each Other",
                                labels={'x': 'Features', 'y': 'Features'})
            fig_corr.update_layout(width=800, height=800)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("""
            <div class='insight-box'>
                <strong>💡 How to Read This:</strong><br>
                • <span style='color: #d62728;'>Dark Red squares</span> = Strong positive correlation (e.g., if Communication Score goes up, Performance Score likely goes up)<br>
                • <span style='color: #1f77b4;'>Dark Blue squares</span> = Strong negative correlation (e.g., if Stress Level goes up, Performance might go down)<br>
                • <span style='color: #7f7f7f;'>Light/White squares</span> = Weak or no correlation<br>
                <br>
                Look for features strongly correlated with <strong>performance_score</strong> and <strong>hiring_decision</strong> 
                - these are the key predictors of interview success!
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Performance Score vs Communication Score")
            st.markdown("""
            <div class='explanation-box'>
                <strong>📊 What This Shows:</strong> This scatter plot reveals the relationship between communication ability 
                and overall interview performance. Each dot represents one candidate, colored by their hiring outcome.
            </div>
            """, unsafe_allow_html=True)
            
            fig_scatter = px.scatter(df_clean, x='communication_score', y='performance_score', 
                                    color='hiring_decision',
                                    hover_data=['candidate_id', 'job_role', 'years_of_experience'],
                                    title="How Communication Skills Impact Overall Performance",
                                    labels={'communication_score': 'Communication Score', 
                                           'performance_score': 'Performance Score'},
                                    color_discrete_sequence=px.colors.qualitative.Set2)
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.6))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown(f"""
            <div class='insight-box'>
                <strong>💡 Key Observations:</strong><br>
                • Each point represents a candidate, with position showing their communication and performance scores<br>
                • Color indicates the hiring decision outcome<br>
                • Hover over points to see individual candidate details (ID, job role, experience)<br>
                • Look for clustering patterns - do hired candidates tend to have higher communication scores?<br>
                <br>
                This visualization helps answer: <strong>"Does better communication lead to better interview performance?"</strong>
            </div>
            """, unsafe_allow_html=True)
            
        with tab3:
            st.subheader("Stress Level Impact on Performance")
            st.markdown("""
            <div class='explanation-box'>
                <strong>📊 What This Shows:</strong> This scatter plot examines how stress levels during the interview affect 
                candidate performance. Understanding this relationship helps identify optimal stress ranges and supports candidates 
                in stress management techniques.
            </div>
            """, unsafe_allow_html=True)
            
            fig_stress = px.scatter(df_clean, x='stress_level', y='performance_score', 
                                   title="Relationship Between Interview Stress and Performance",
                                   labels={'stress_level': 'Stress Level (0-100)', 
                                          'performance_score': 'Performance Score'},
                                   color_discrete_sequence=['#e74c3c'])
            fig_stress.update_traces(marker=dict(size=8, opacity=0.5))
            st.plotly_chart(fig_stress, use_container_width=True)
            
            # Calculate correlation
            stress_perf_corr = df_clean['stress_level'].corr(df_clean['performance_score'])
            st.markdown(f"""
            <div class='insight-box'>
                <strong>💡 Analysis:</strong><br>
                • <strong>Correlation Coefficient:</strong> {stress_perf_corr:.3f}<br>
                • {'Negative correlation: Higher stress tends to reduce performance.' if stress_perf_corr < -0.1 else 'Positive correlation: Some stress may enhance performance.' if stress_perf_corr > 0.1 else 'Weak correlation: Stress has minimal impact on performance.'}<br>
                • The trendline shows the general pattern across all candidates<br>
                <br>
                <strong>Practical Implication:</strong> {'Candidates should practice stress management techniques before interviews.' if stress_perf_corr < -0.1 else 'Moderate stress might actually help candidates stay alert and engaged.' if stress_perf_corr > 0.1 else 'Focus on other factors beyond stress for better performance.'}
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Experience Level vs Hiring Decision")
            st.markdown("""
            <div class='explanation-box'>
                <strong>📊 What This Shows:</strong> Box plots compare years of experience across different hiring outcomes. 
                This helps answer: "Do more experienced candidates get hired more often?"
            </div>
            """, unsafe_allow_html=True)
            
            fig_box = px.box(df_clean, x='hiring_decision', y='years_of_experience', 
                            color='hiring_decision', 
                            title="Does Experience Matter? Experience Distribution by Hiring Outcome",
                            labels={'hiring_decision': 'Hiring Decision', 
                                   'years_of_experience': 'Years of Professional Experience'},
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistical summary
            exp_by_decision = df_clean.groupby('hiring_decision')['years_of_experience'].agg(['mean', 'median', 'std'])
            st.markdown(f"""
            <div class='insight-box'>
                <strong>💡 What Box Plots Tell Us:</strong><br>
                • <strong>Middle line</strong> in each box = Median experience for that hiring category<br>
                • <strong>Box edges</strong> = 25th and 75th percentiles (middle 50% of candidates)<br>
                • <strong>Whiskers</strong> = Full range excluding outliers<br>
                • <strong>Dots</strong> = Outliers (unusually high/low experience for that outcome)<br>
                <br>
                Compare the medians across categories to see if experience correlates with hiring success!
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Education Level Breakdown")
            st.markdown("""
            <div class='explanation-box'>
                <strong>📊 What This Shows:</strong> This pie chart displays the educational background distribution of all candidates. 
                Understanding educational diversity helps assess whether the dataset represents various education levels fairly.
            </div>
            """, unsafe_allow_html=True)
            
            edu_counts = df_clean['education'].value_counts()
            fig_pie = px.pie(df_clean, names='education', 
                            title="Educational Background of Interview Candidates",
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown(f"""
            <div class='insight-box'>
                <strong>💡 Education Distribution:</strong><br>
                {edu_counts.to_frame().to_html(index=False)}<br>
                This breakdown shows whether certain education levels are over/under-represented in the candidate pool.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Dataset not found. Please ensure 'interview_performance_data.csv' is in the data folder.")

# --- 3. AI PERFORMANCE PREDICTION ---
elif page == "AI Performance Prediction":
    st.title("🎯 AI Candidate Evaluation")
    st.markdown("---")
    
    st.markdown("""
    <div class='explanation-box'>
        <strong>🤖 How It Works:</strong> Our trained Random Forest machine learning model analyzes candidate inputs using 
        <strong>16 key features</strong> to predict hiring outcomes. The model learned patterns from thousands of historical 
        interviews, identifying which combinations of behavioral traits, communication skills, and psychological factors lead 
        to successful hiring decisions.
    </div>
    """, unsafe_allow_html=True)
    
    if model is not None:
        st.sidebar.header("📝 Input Candidate Metrics")
        st.sidebar.markdown("""
        <div class='tip-box'>
            <strong>💡 Tip:</strong> Enter realistic values based on actual interview observations. 
            The more accurate the inputs, the more reliable the prediction!
        </div>
        """, unsafe_allow_html=True)
        
        # Create input fields based on the 17 features used in the model
        age = st.sidebar.slider("Age", 18, 65, 30, help="Candidate's current age in years")
        exp = st.sidebar.slider("Years of Experience", 0, 40, 5, help="Total years of professional work experience")
        comm = st.sidebar.slider("Communication Score", 0, 150, 80, help="Overall communication effectiveness (0-150): includes clarity, articulation, and coherence")
        vocab = st.sidebar.slider("Vocabulary Richness", 0.0, 1.0, 0.5, help="Word usage diversity (0-1): higher values indicate more varied vocabulary")
        filler = st.sidebar.slider("Filler Words per Min", 0, 20, 5, help="Frequency of hesitation words like 'um', 'uh', 'like' per minute")
        conf = st.sidebar.slider("Confidence Score", 0, 100, 75, help="AI-assessed confidence level (0-100) from voice tone and body language")
        eye = st.sidebar.slider("Eye Contact Score", 0, 100, 80, help="Percentage of time maintaining appropriate eye contact with camera")
        stress = st.sidebar.slider("Stress Level", 0, 100, 30, help="Physiological stress indicators (0-100): lower is generally better")
        resp = st.sidebar.slider("Avg Response Time (sec)", 1.0, 15.0, 4.0, help="Average time taken to respond to questions (optimal: 3-6 seconds)")
        dur = st.sidebar.slider("Interview Duration (min)", 15, 60, 30, help="Total length of the interview session in minutes")
        sent = st.sidebar.slider("Sentiment Score", -1.0, 1.0, 0.2, help="Emotional tone detected in speech (-1 to +1): positive is better")
        
        # Categorical inputs
        job_role = st.sidebar.selectbox("Job Role", 
                                       ["Software Engineer", "Data Scientist", "Marketing Analyst", 
                                        "Sales Executive", "HR Manager"],
                                       help="Select the position the candidate applied for")
        edu = st.sidebar.selectbox("Education", 
                                  ["Bachelor's", "Master's", "PhD", "Associate"],
                                  help="Highest educational qualification")
        platform = st.sidebar.selectbox("Platform", 
                                       ["Zoom", "Teams", "HireVue", "Google Meet"],
                                       help="Video conferencing platform used for the interview")
        posture = st.sidebar.selectbox("Posture", 
                                      ["Confident", "Neutral", "Slouching"],
                                      help="Observed body posture during the interview")
        
        st.subheader("📊 Feature Importance Guide")
        
        # Communication & Language Metrics
        st.markdown("#### 🗣️ Communication & Language:")
        st.markdown("""
        - **Communication Score:** Overall verbal effectiveness
        - **Vocabulary Richness:** Word diversity and sophistication
        - **Filler Words:** Hesitation frequency (lower = better)
        - **Sentiment:** Emotional positivity in responses
        """)
        
        # Behavioral & Non-Verbal Metrics
        st.markdown("#### 🎭 Behavioral & Non-Verbal:")
        st.markdown("""
        - **Confidence:** Self-assurance level from AI analysis
        - **Eye Contact:** Engagement indicator
        - **Posture:** Body language classification
        """)
        
        # Psychological Metrics
        st.markdown("#### 🧠 Psychological:")
        st.markdown("""
        - **Stress Level:** Anxiety/tension indicators
        - **Response Time:** Thoughtfulness vs hesitation
        """)
        
        if st.button("🚀 Predict Hiring Decision", type="primary"):
            with st.spinner("🤖 AI is analyzing candidate profile..."):
                # Prepare input array (must match the 16 features used during training)
                input_data = np.zeros((1, 16))
                input_data[0, 0] = age
                input_data[0, 1] = exp
                input_data[0, 2] = comm
                input_data[0, 3] = vocab
                input_data[0, 4] = filler
                input_data[0, 5] = conf
                input_data[0, 6] = eye
                input_data[0, 7] = stress
                input_data[0, 8] = resp
                input_data[0, 9] = dur
                input_data[0, 10] = sent
                # Categorical encodings
                job_role_map = {"Software Engineer": 0, "Data Scientist": 1, "Marketing Analyst": 2, "Sales Executive": 3, "HR Manager": 4}
                edu_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2, "Associate": 3}
                platform_map = {"Zoom": 0, "Teams": 1, "HireVue": 2, "Google Meet": 3}
                posture_map = {"Confident": 0, "Neutral": 1, "Slouching": 2}
                input_data[0, 11] = job_role_map.get(job_role, 0)
                input_data[0, 12] = edu_map.get(edu, 0)
                input_data[0, 13] = platform_map.get(platform, 0)
                input_data[0, 14] = posture_map.get(posture, 0)
                input_data[0, 15] = 0  # Placeholder for feature alignment
                
                prediction = model.predict(input_data)
                
                # Map prediction back to label
                decisions = {0: "Hired", 1: "On Hold", 2: "Rejected", 3: "Second Round"}
                result = decisions.get(prediction[0], "Unknown")
                
                st.subheader("🎯 Prediction Result")
                
                # Display result with appropriate styling
                if result == "Hired":
                    st.balloons()
                    st.success(f"**✅ Recommendation: {result}**\n\nThe AI model predicts this candidate should be hired based on their strong performance across multiple dimensions.")
                elif result == "Second Round":
                    st.info(f"**🔄 Recommendation: {result}**\n\nThe AI suggests further evaluation. The candidate shows potential but needs additional assessment.")
                elif result == "On Hold":
                    st.warning(f"**⏸️ Recommendation: {result}**\n\nThe AI indicates uncertainty. Consider comparing with other candidates or gathering more data.")
                else:
                    st.error(f"**❌ Recommendation: {result}**\n\nThe AI predicts this candidate doesn't meet the requirements based on the evaluated metrics.")
                
                # Show a radar chart of the candidate's strengths
                st.markdown("---")
                st.subheader("📊 Candidate Strength Profile")
                
                st.markdown("""
                <div class='explanation-box'>
                    <strong>🕵️ Radar Chart Analysis:</strong> This visualization shows the candidate's relative strengths 
                    across five key dimensions. Larger area indicates stronger overall profile. Compare against ideal ranges 
                    to identify improvement areas.
                </div>
                """, unsafe_allow_html=True)
                
                categories = ['Communication', 'Confidence', 'Eye Contact', 'Stress Mgmt', 'Positive Outlook']
                values = [
                    min(comm/1.5, 100),  # Normalize to 0-100
                    conf,                 # Already 0-100
                    eye,                  # Already 0-100
                    100-stress,          # Invert stress (higher is better)
                    (sent+1)*50          # Convert -1 to +1 into 0-100
                ]
                
                fig_radar = px_go.Figure()
                fig_radar.add_trace(px_go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Candidate Profile',
                    line_color='#3498db',
                    fillcolor='rgba(52, 152, 219, 0.5)'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    title="Candidate's Behavioral & Communication Strengths",
                    width=600,
                    height=600
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Detailed breakdown
                st.markdown("### 📋 Detailed Analysis Breakdown")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Strengths**")
                    if comm > 80:
                        st.write("• Excellent communication skills")
                    if conf > 75:
                        st.write("• Strong confidence level")
                    if eye > 75:
                        st.write("• Great eye contact")
                    if stress < 30:
                        st.write("• Good stress management")
                    if sent > 0.3:
                        st.write("• Positive emotional tone")
                    if vocab > 0.6:
                        st.write("• Rich vocabulary")
                    
                    if all([comm <= 80, conf <= 75, eye <= 75, stress >= 30, sent <= 0.3, vocab <= 0.6]):
                        st.write("Review all metrics for improvement areas")
                
                with col2:
                    st.markdown("**⚠️ Areas for Improvement**")
                    if comm <= 60:
                        st.write("• Work on communication clarity")
                    if conf <= 60:
                        st.write("• Build more confidence")
                    if eye <= 60:
                        st.write("• Improve eye contact")
                    if stress >= 60:
                        st.write("• Practice stress reduction techniques")
                    if sent <= -0.2:
                        st.write("• Develop more positive framing")
                    if filler > 10:
                        st.write("• Reduce filler word usage")
                    
                    if all([comm > 60, conf > 60, eye > 60, stress < 60, sent > -0.2, filler <= 10]):
                        st.write("Well-rounded profile across all metrics!")
                
                st.markdown("""
                <div class='insight-box'>
                    <strong>💡 What This Means:</strong> The radar chart above visualizes where this candidate excels and where 
                    there's room for growth. HR professionals can use this breakdown to provide targeted feedback and coaching.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Model file not found. Please ensure 'model.pkl' is in the models folder.")

# --- 4. DOWNLOAD REPORTS ---
elif page == "Download Reports":
    st.title("📥 Download Resources")
    st.markdown("---")
    
    st.markdown("""
    <div class='explanation-box'>
        <strong>📚 Available Downloads:</strong> Access the complete dataset and comprehensive analysis report from this 
        Digital Interview Performance Scoring project. These resources include raw data, statistical analysis, visualizations, and key insights.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if data is available
    if df is not None:
        # Generate buttons for both downloads
        tab1, tab2 = st.tabs(["📊 Dataset", "📄 Analysis Report"])
        
        with tab1:
            st.subheader("Download Raw Dataset (CSV)")
            st.markdown("""
            **What's Included:**
            - Complete interview performance dataset
            - 7,501 candidate records
            - 17 features including behavioral metrics and outcomes
            - Ready for CSV export for your own analysis
            
            **Metrics Included:**
            - Demographics: Age, Years of Experience
            - Communication: Communication Score, Vocabulary Richness, Filler Words, Sentiment
            - Behavioral: Confidence, Eye Contact, Posture
            - Psychological: Stress Level, Response Time
            - Outcomes: Performance Score, Hiring Decision
            """)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Dataset",
                data=csv,
                file_name='interview_performance_data.csv',
                mime='text/csv',
                help="Download the complete dataset in CSV format",
                type="primary",
                use_container_width=True
            )
            st.info(f"File size: ~{(len(csv)/1024):.1f} KB | Records: {len(df):,}")
            
            st.markdown("""
            ### Key Statistics
            """)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Candidates", f"{len(df):,}")
            with col2:
                st.metric("Features", "17 metrics")
            with col3:
                st.metric("Avg Performance", f"{df['performance_score'].mean():.2f}")
            with col4:
                st.metric("Hiring Rate", f"{(df['hiring_decision'] == 'Hired').mean()*100:.1f}%")
        
        with tab2:
            st.subheader("Download Comprehensive Analysis Report (HTML)")
            st.markdown("""
            **What's Included:**
            - Executive summary with key metrics
            - Dataset overview and feature descriptions
            - Statistical analysis and findings
            - Hiring decision distribution
            - Correlation analysis insights
            - Machine learning model details
            - Professional recommendations for HR
            - Beautiful formatting with charts and tables
            
            **Report Format:**
            - Interactive HTML document
            - Professional design with gradient headers
            - Metric cards and info boxes
            - Print-friendly layout
            - Can be converted to PDF using browser print function
            """)
            
            # Generate the HTML report
            html_report = generate_html_report(df)
            
            st.download_button(
                label="📥 Download HTML Report",
                data=html_report,
                file_name='interview_performance_analysis_report.html',
                mime='text/html',
                help="Download the comprehensive analysis report in HTML format",
                type="primary",
                use_container_width=True
            )
            
            st.info("""
            **💡 How to Save as PDF:**
            1. Open the downloaded HTML file in your web browser
            2. Press Ctrl+P (Windows) or Cmd+P (Mac) to print
            3. Select 'Save as PDF' as the destination
            4. Click Save to create a PDF version
            
            The report includes all key findings, statistics, and insights from the analysis!
            """)
            
            st.markdown("""
            ### Report Sections
            - 📋 Executive Summary
            - 📊 Dataset Overview  
            - 📈 Key Findings
            - 🔍 Correlation Analysis
            - 💡 HR Recommendations
            - 🤖 ML Model Details
            """)
    else:
        st.error("Dataset not found. Please ensure 'interview_performance_data.csv' is in the data folder.")
    
    st.markdown("---")
    
    st.markdown("""
    <div class='insight-box'>
        <strong>💡 Usage Note:</strong> These resources are provided for educational and research purposes. 
        They demonstrate the application of machine learning to HR analytics and interview performance evaluation.
    </div>
    """, unsafe_allow_html=True)
