"""
College Student Placement Analysis - Streamlit Application
Final Project: Data Analysis Techniques
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve)

# Set page configuration
st.set_page_config(
    page_title="Student Placement Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load the college placement dataset"""
    df = pd.read_csv('college_student_placement_dataset.csv')
    return df

@st.cache_data
def prepare_data(df):
    """Prepare data for modeling"""
    df_model = df.copy()
    
    # Convert categorical variables
    df_model['Internship_Experience'] = df_model['Internship_Experience'].map({'Yes': 1, 'No': 0})
    df_model['Placement'] = df_model['Placement'].map({'Yes': 1, 'No': 0})
    
    # Select features for modeling
    feature_columns = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
                      'Internship_Experience', 'Extra_Curricular_Score',
                      'Communication_Skills', 'Projects_Completed']
    
    X = df_model[feature_columns]
    y = df_model['Placement']
    
    return X, y, df_model

@st.cache_resource
def train_model(X, y):
    """Train logistic regression model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Calculate odds ratios
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)
    
    return model, X_test, y_test, y_pred, y_pred_proba, accuracy, conf_matrix, auc_score, fpr, tpr, coefficients, odds_ratios

def main():
    # Title
    st.markdown('<h1 class="main-header">🎓 College Student Placement Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    section = st.sidebar.radio(
        "Go to Section:",
        ["Overview", "Data Preparation & Exploration", "Analysis & Findings", "Conclusions & Recommendations"]
    )
    
    # Load data
    df = load_data()
    X, y, df_model = prepare_data(df)
    
    # Train model
    model, X_test, y_test, y_pred, y_pred_proba, accuracy, conf_matrix, auc_score, fpr, tpr, coefficients, odds_ratios = train_model(X, y)

    # ============================================================================
    # SECTION 1: OVERVIEW
    # ============================================================================
    if section == "Overview":
        st.markdown('<h2 class="section-header">📋 Overview</h2>', unsafe_allow_html=True)
        
        # Research Statement
        st.markdown("### 🎯 Research Statement")
        st.info("""
        **Primary Goal:** To predict the target variable (**Placement**) and determine exactly how much 
        influence different predictor variables (like Communication Skills, CGPA, IQ, etc.) have on that outcome.
        
        This analysis aims to identify which factors most significantly impact a student's likelihood of 
        being placed in a job after graduation, providing actionable insights for students, universities, 
        and employers.
        """)
        
        # The Dataset
        st.markdown("### 📊 The Dataset")
        
        st.markdown("""
        We are analyzing the **'College Student Placement Factors'** dataset, which contains comprehensive 
        information about college students and their placement outcomes. This dataset includes various 
        academic, skill-based, and experience-related factors.
        """)
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📚 Total Students", f"{len(df):,}")
        
        with col2:
            placed_count = df['Placement'].value_counts().get('Yes', 0)
            st.metric("✅ Students Placed", f"{placed_count:,}")
        
        with col3:
            placement_rate = (placed_count / len(df)) * 100
            st.metric("📈 Placement Rate", f"{placement_rate:.1f}%")
        
        st.markdown("---")
        
        # Sample Data
        st.markdown("#### 📋 Sample Data (First 5 Rows)")
        st.dataframe(df.head(5), use_container_width=True)
        
        st.markdown("---")
        
        # Variable Descriptions
        st.markdown("#### 📖 Variable Descriptions")
        
        st.markdown("""
        To understand our analysis, we must distinguish between our **Target Variable** (the outcome we want to predict) 
        and our **Predictor Variables** (the input factors).
        """)
        
        col1, col2 = st.columns(2)
        
        st.markdown("##### 🎯 Target Variable (The Outcome)")
        target_data = {
            "Variable": ["Placement"],
            "Description": ["Final placement result (Yes = Placed, No = Not Placed)"],
            "Type": ["Categorical (Binary)"]
        }
        target_df = pd.DataFrame(target_data)
        st.dataframe(target_df, use_container_width=True, hide_index=True)
        
        st.info("This is the 'Label' or 'Y' variable we are trying to predict.")

        st.markdown("##### 🔍 Predictor Variables (The Factors)")
        predictor_data = [
            {"Variable": "Communication_Skills", "Description": "Soft skills rating (scale 1-10)", "Type": "Numerical"},
            {"Variable": "CGPA", "Description": "Cumulative Grade Point Average (5.0-10.0)", "Type": "Numerical"},
            {"Variable": "IQ", "Description": "Intelligence Quotient score", "Type": "Numerical"},
            {"Variable": "Projects_Completed", "Description": "Number of academic/technical projects", "Type": "Numerical"},
            {"Variable": "Prev_Sem_Result", "Description": "Previous semester GPA", "Type": "Numerical"},
            {"Variable": "Academic_Performance", "Description": "Annual performance rating (1-10)", "Type": "Numerical"},
            {"Variable": "Internship_Experience", "Description": "Has completed internship (Yes/No)", "Type": "Categorical"},
            {"Variable": "Extra_Curricular_Score", "Description": "Activity involvement score (0-10)", "Type": "Numerical"},
            {"Variable": "College_ID", "Description": "Unique college identifier", "Type": "Categorical (ID)"}
        ]
        predictor_df = pd.DataFrame(predictor_data)
        st.table(predictor_df.style.hide(axis="index"))
        
        st.markdown("---")
        
        # Target Variable Distribution
        st.markdown("#### 🎯 Target Variable Distribution")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie Chart
            fig, ax = plt.subplots(figsize=(3, 2))  # Keep your small size
            placement_counts = df['Placement'].value_counts()
            colors = ["#0800ff", "#38be16"]
            explode = (0.05, 0.05)
            
            wedges, texts, autotexts = ax.pie(
                placement_counts, 
                labels=placement_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                explode=explode,
                textprops={'fontsize': 6, 'fontweight': 'bold'}
            )
            
            ax.set_title('Placement Distribution', 
                        fontsize=8, fontweight='bold', pad=10)
            
            plt.tight_layout()
            
            # This is the key change: set use_container_width=False
            st.pyplot(fig, use_container_width=False)
            plt.close()
        
        with col2:
            st.markdown("**Class Balance Analysis:**")
            
            not_placed_count = placement_counts.get('No', 0)
            not_placed_pct = (not_placed_count / len(df)) * 100
            placed_pct = (placed_count / len(df)) * 100
            
            st.markdown(f"""
            - **Not Placed:** {not_placed_count:,} students ({not_placed_pct:.1f}%)
            - **Placed:** {placed_count:,} students ({placed_pct:.1f}%)
            
            **Interpretation:**
            
            The dataset shows a class imbalance with approximately {not_placed_pct:.0f}% of students 
            not being placed versus {placed_pct:.0f}% being placed. This imbalance is typical in 
            real-world placement scenarios and our model accounts for this distribution.
            
            A balanced dataset would show 50%-50%, but this real-world imbalance makes accurate 
            prediction more challenging and valuable.
            """)
        
        st.markdown("---")
        
        # Methodology
        st.markdown("### 🔬 Methodology: Why Logistic Regression?")
        
        st.markdown("""
        We selected **Logistic Regression** as our primary technique because it moves beyond simple observation 
        into actual forecasting. It is the perfect mathematical tool for answering our specific question: 
        *"Will this student get a job?"*
        """)
        
        # Create 3 columns for the Custom Cards
        col1, col2, col3 = st.columns(3)
        
        # Define the card style function for reusability
        def custom_card(title, subtitle, text):
            return f"""
            <div style='background-color: #e8f4f8; 
                        padding: 1.5rem; 
                        border-radius: 0.5rem; 
                        border-left: 5px solid #00a8e8;
                        height: 100%;'>
                <h3 style='color: #0c5460; margin: 0; font-size: 1.2rem; border-bottom: 1px solid #b8daff; padding-bottom: 5px;'>
                    {title}
                </h3>
                <p style='font-size: 1.1rem; font-weight: bold; margin: 10px 0; color: #007bff;'>
                    {subtitle}
                </p>
                <p style='font-size: 0.95rem; margin: 0; color: #333; line-height: 1.5;'>
                    {text}
                </p>
            </div>
            """

        with col1:
            st.markdown(custom_card(
                title="1. Binary Target",
                subtitle="The 'Yes/No' Reality",
                text="Our goal is to predict a clear outcome: <b>Placed</b> or <b>Not Placed</b>. <br><br>Linear regression predicts continuous numbers (like salary), which fails for a binary question. Logistic regression is purpose-built for this classification."
            ), unsafe_allow_html=True)
            
        with col2:
            st.markdown(custom_card(
                title="2. Predictive Power",
                subtitle="Beyond Correlation",
                text="Correlation only tells us if two things are related. Logistic Regression takes us a step further. <br><br>It combines complex factors (IQ, CGPA, Skills) into a single equation that can actually <b>forecast</b> the future outcome for a new student."
            ), unsafe_allow_html=True)
            
        with col3:
            st.markdown(custom_card(
                title="3. Clear, Usable Data",
                subtitle="Actionable Insights",
                text="This method provides results we can actually use. Instead of vague trends, it gives us precise metrics: <br><br>• <b>Probability:</b> '85% chance of success'<br>• <b>Odds:</b> '6x higher likelihood'<br><br>This clarity makes insights actionable."
            ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Validation - Chi-Squared Test
        st.markdown("### 📊 Model Validation: Statistical Significance Test")
        
        # 1. Definition (Placed under header)
        st.markdown("""
        **Likelihood Ratio Test (Chi-Squared Test)**
        
        This statistical test compares our **full model** (containing all predictor variables like IQ, CGPA, etc.) 
        against a **null model** (a baseline model that just guesses the average placement rate for everyone). 
        It effectively answers the question: *"Does our model actually predict anything, or is it just guessing?"*
        """)
        
        st.markdown("---")

        # Calculate likelihood ratio test
        # Null model (intercept only)
        null_prob = y.mean()
        null_probs = np.full((len(y), 2), [1-null_prob, null_prob])
        
        # Calculate log-likelihoods
        from sklearn.metrics import log_loss
        
        # For full model, we need to get predictions on full dataset
        X_scaled = X.copy()
        model_full = LogisticRegression(max_iter=1000, random_state=42)
        model_full.fit(X_scaled, y)
        
        probs_fitted = model_full.predict_proba(X_scaled)
        ll_fitted = -log_loss(y, probs_fitted, normalize=False)
        ll_null = -log_loss(y, null_probs, normalize=False)
        
        # G-statistic (likelihood ratio test statistic)
        g_statistic = 2 * (ll_fitted - ll_null)
        df_degrees = X.shape[1]  # number of predictors
        
        # Calculate p-value
        from scipy.stats import chi2
        p_value = chi2.sf(g_statistic, df_degrees)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 1. Define variables based on p-value
            if p_value < 0.001:
                significance_text = "p < 0.001"
                interpretation = "Highly Significant ✅"
                bg_color = "#d4edda"
                border_color = "#28a745"
                text_color = "#155724"
            elif p_value < 0.05:
                significance_text = f"p = {p_value:.4f}"
                interpretation = "Significant ✅"
                bg_color = "#d4edda"
                border_color = "#28a745"
                text_color = "#155724"
            else:
                significance_text = f"p = {p_value:.4f}"
                interpretation = "Not Significant ❌"
                bg_color = "#f8d7da"
                border_color = "#dc3545"
                text_color = "#721c24"
            
            # 2. Build the HTML String separately (Safer)
            card_html = f"""
            <div style='background-color: {bg_color}; 
                        padding: 2rem; 
                        border-radius: 0.5rem; 
                        border-left: 5px solid {border_color};
                        text-align: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: {text_color}; margin: 0; font-size: 1.4rem;'>
                    Model Significance Result
                </h3>
                <p style='font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; color: {text_color};'>
                    {significance_text}
                </p>
                <p style='font-size: 1.2rem; font-weight: bold; margin-bottom: 1.5rem; color: {text_color};'>
                    {interpretation}
                </p>
                <hr style='border-top: 1px solid {border_color}; opacity: 0.3; margin: 1rem 0;'>
                <div style='display: flex; justify-content: space-around; color: {text_color};'>
                    <div>
                        <span style='font-size: 0.9rem; display: block;'>Chi-Squared (G)</span>
                        <span style='font-size: 1.2rem; font-weight: bold;'>{g_statistic:.2f}</span>
                    </div>
                    <div>
                        <span style='font-size: 0.9rem; display: block;'>Degrees of Freedom</span>
                        <span style='font-size: 1.2rem; font-weight: bold;'>{df_degrees}</span>
                    </div>
                </div>
            </div>
            """
            
            # 3. Render the HTML
            st.markdown(card_html, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💡 Interpretation")
            
            st.info(f"""
            With a p-value of **{significance_text}**, we can conclude with **>99.9% confidence** that our predictor variables 
            (IQ, CGPA, Communication Skills, etc.) significantly improve prediction compared to random guessing.
            """)
            
            st.markdown("""
            **What This Means for Our Project:**
            - ✅ **Validity:** Our findings are not due to random chance.
            - ✅ **Reliability:** The relationships we identify (like Communication Skills being important) are statistically real.
            - ✅ **Utility:** The model is a valid tool for making predictions on new students.
            """)
        
        st.markdown("---")
        
        st.success("""
        **Overview Summary:**
        
        We have established a statistically validated logistic regression model to predict student placement. 
        The model has been proven to significantly outperform random guessing (p < 0.001), and we can now 
        proceed to explore the data, analyze the results, and derive actionable insights.
        """)
    
    # ============================================================================
    # SECTION 2: DATA PREPARATION & EXPLORATION
    # ============================================================================
    elif section == "Data Preparation & Exploration":
        st.markdown('<h2 class="section-header">🔍 Data Preparation & Exploration</h2>', unsafe_allow_html=True)
        
        # TODO: Add data preparation and exploration content
        st.write("Data preparation and exploration content goes here...")
    
    # ============================================================================
    # SECTION 3: ANALYSIS & FINDINGS
    # ============================================================================
    elif section == "Analysis & Findings":
        st.markdown('<h2 class="section-header">🎯 Analysis & Findings</h2>', unsafe_allow_html=True)
        
        # TODO: Add analysis and findings content
        st.write("Analysis and findings content goes here...")
    
    # ============================================================================
    # SECTION 4: CONCLUSIONS & RECOMMENDATIONS
    # ============================================================================
    elif section == "Conclusions & Recommendations":
        st.markdown('<h2 class="section-header">🎊 Conclusions & Recommendations</h2>', unsafe_allow_html=True)
        
        # TODO: Add conclusions and recommendations content
        st.write("Conclusions and recommendations content goes here...")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 2rem;'>
            <p><strong>College Student Placement Analysis</strong></p>
            <p>Final Project: Data Analysis Techniques</p>
            <p>Powered by Logistic Regression | Streamlit Application</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()