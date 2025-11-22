import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Interactive Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #4f46e5;
        border-bottom: 3px solid #4f46e5;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #dbeafe;
        border-left: 5px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio(
    "Go to Section:",
    ["Overview", "Data Exploration & Preparation", "Analysis & Insights", "Conclusions & Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Course:** [Your Course Name]\n\n**Student:** [Your Name]\n\n**Date:** [Presentation Date]")

# Main header
st.markdown('<p class="main-header">Interactive Data Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Exploring patterns and insights through data</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SECTION 1: OVERVIEW
# ============================================================================
if section == "Overview":
    st.markdown('<p class="section-header">📋 Dataset Overview</p>', unsafe_allow_html=True)

    # Research Question
    st.markdown("""
        <div class="info-box">
            <h3>🔍 Research Question</h3>
            <p>[Your research question will be displayed here. For example: "What factors most significantly 
            influence customer churn in subscription-based services?"]</p>
        </div>
    """, unsafe_allow_html=True)

    # Dataset Information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Dataset Information")
        st.write("""
        - **Source:** [Dataset source - e.g., Kaggle, UCI, company data]
        - **Records:** [Number of rows]
        - **Features:** [Number of columns]
        - **Time Period:** [Date range if applicable]
        - **Target Variable:** [Your dependent variable]
        """)

    with col2:
        st.subheader("🔬 Analysis Technique")
        st.write("""
        [Description of your selected analysis technique]

        **Examples:**
        - Linear/Logistic Regression
        - K-Means Clustering
        - Decision Trees/Random Forest
        - Time Series Analysis
        - Principal Component Analysis (PCA)
        """)

    # Dataset Structure
    st.subheader("📊 Dataset Structure")

    # Sample dataframe structure (replace with your actual data)
    sample_structure = pd.DataFrame({
        'Column Name': ['column_1', 'column_2', 'column_3', 'column_4', 'column_5'],
        'Data Type': ['int64', 'float64', 'object', 'datetime64', 'bool'],
        'Description': [
            'Description of column 1',
            'Description of column 2',
            'Description of column 3',
            'Description of column 4',
            'Description of column 5'
        ]
    })

    st.dataframe(sample_structure, use_container_width=True)

    # Sample data preview
    st.subheader("👀 Data Preview")
    st.write("First few rows of the dataset:")

    # Create sample data (replace with your actual data)
    sample_data = pd.DataFrame({
        'column_1': [1, 2, 3, 4, 5],
        'column_2': [10.5, 20.3, 15.7, 30.2, 25.8],
        'column_3': ['A', 'B', 'A', 'C', 'B'],
        'column_4': pd.date_range('2024-01-01', periods=5),
        'column_5': [True, False, True, True, False]
    })

    st.dataframe(sample_data, use_container_width=True)

# ============================================================================
# SECTION 2: DATA EXPLORATION & PREPARATION
# ============================================================================
elif section == "Data Exploration & Preparation":
    st.markdown('<p class="section-header">🔍 Data Exploration & Preparation</p>', unsafe_allow_html=True)

    # Data Cleaning Steps
    st.markdown("""
        <div class="warning-box">
            <h3>🧹 Data Cleaning Steps</h3>
            <ul>
                <li><strong>Handling Missing Values:</strong> [Describe your approach - imputation, removal, etc.]</li>
                <li><strong>Outlier Detection:</strong> [Your method - IQR, Z-score, domain knowledge]</li>
                <li><strong>Data Transformations:</strong> [Scaling, normalization, encoding, etc.]</li>
                <li><strong>Feature Engineering:</strong> [New features created, feature selection]</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Data Distribution")
        # Sample histogram (replace with your actual data)
        sample_data = np.random.normal(100, 15, 1000)
        fig1 = px.histogram(x=sample_data, nbins=30,
                            title="Distribution of [Variable Name]",
                            labels={'x': 'Value', 'y': 'Frequency'})
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("🔥 Correlation Heatmap")
        # Sample correlation matrix (replace with your actual data)
        corr_data = np.random.rand(5, 5)
        corr_df = pd.DataFrame(corr_data,
                               columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'],
                               index=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])
        fig2 = px.imshow(corr_df,
                         title="Feature Correlation Matrix",
                         color_continuous_scale='RdBu_r',
                         aspect='auto')
        st.plotly_chart(fig2, use_container_width=True)

    # Missing Data
    st.subheader("❓ Missing Data Analysis")
    missing_data = pd.DataFrame({
        'Column': ['column_1', 'column_2', 'column_3', 'column_4', 'column_5'],
        'Missing Count': [0, 15, 3, 0, 8],
        'Missing Percentage': [0.0, 1.5, 0.3, 0.0, 0.8]
    })

    fig3 = px.bar(missing_data, x='Column', y='Missing Percentage',
                  title="Missing Data by Column",
                  labels={'Missing Percentage': 'Missing %'})
    st.plotly_chart(fig3, use_container_width=True)

    # Statistical Summary
    st.subheader("📊 Statistical Summary")
    summary_data = pd.DataFrame({
        'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
        'Variable_1': [1000, 50.2, 15.3, 10.5, 40.1, 49.8, 60.3, 95.7],
        'Variable_2': [1000, 100.5, 25.8, 45.2, 80.3, 98.7, 120.4, 180.2]
    })
    st.dataframe(summary_data, use_container_width=True)

# ============================================================================
# SECTION 3: ANALYSIS & INSIGHTS
# ============================================================================
elif section == "Analysis & Insights":
    st.markdown('<p class="section-header">📊 Analysis & Insights</p>', unsafe_allow_html=True)

    # Interactive Filters
    st.subheader("🎛️ Interactive Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_category = st.selectbox(
            "Filter by Category:",
            ["All Data", "Category 1", "Category 2", "Category 3"]
        )

    with col2:
        threshold_value = st.slider(
            "Adjust Threshold:",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )

    with col3:
        date_range = st.date_input(
            "Select Date Range:",
            value=(pd.to_datetime("2024-01-01"), pd.to_datetime("2024-12-31"))
        )

    st.markdown("---")

    # Main Visualization
    st.subheader("📈 Main Analysis Visualization")

    # Sample scatter plot (replace with your actual analysis)
    np.random.seed(42)
    x_data = np.random.normal(50, 10, 200)
    y_data = 2 * x_data + np.random.normal(0, 15, 200)

    fig4 = px.scatter(x=x_data, y=y_data,
                      title="[Your Analysis Title - e.g., Feature Relationship]",
                      labels={'x': 'Independent Variable', 'y': 'Dependent Variable'},
                      trendline="ols")
    fig4.update_layout(height=500)
    st.plotly_chart(fig4, use_container_width=True)

    # Key Insights
    st.subheader("💡 Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="insight-box">
                <h4>🎯 Key Pattern 1</h4>
                <p>[Description of first major insight or pattern discovered in your analysis]</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="insight-box">
                <h4>📊 Key Pattern 2</h4>
                <p>[Description of second major insight or trend identified]</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="insight-box">
                <h4>⚠️ Anomaly Detected</h4>
                <p>[Description of any interesting anomalies or outliers found]</p>
            </div>
        """, unsafe_allow_html=True)

    # Secondary Visualizations
    st.subheader("📊 Additional Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Sample bar chart
        categories = ['Group A', 'Group B', 'Group C', 'Group D']
        values = [23, 45, 32, 51]
        fig5 = px.bar(x=categories, y=values,
                      title="Comparison Across Groups",
                      labels={'x': 'Groups', 'y': 'Metric Value'})
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        # Sample pie chart
        fig6 = px.pie(values=values, names=categories,
                      title="Distribution by Category")
        st.plotly_chart(fig6, use_container_width=True)

    # Model Performance Metrics (if applicable)
    st.subheader("🎯 Model Performance")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Accuracy", "85.3%", "2.1%")

    with metric_col2:
        st.metric("Precision", "82.7%", "-1.5%")

    with metric_col3:
        st.metric("Recall", "88.1%", "3.2%")

    with metric_col4:
        st.metric("F1-Score", "85.3%", "0.8%")

# ============================================================================
# SECTION 4: CONCLUSIONS & RECOMMENDATIONS
# ============================================================================
elif section == "Conclusions & Recommendations":
    st.markdown('<p class="section-header">🎯 Conclusions & Recommendations</p>', unsafe_allow_html=True)

    # Main Takeaways
    st.markdown("""
        <div class="info-box">
            <h3>🏆 Main Takeaways</h3>
            <ol>
                <li><strong>[First Major Conclusion]</strong> - Brief explanation of your first key finding</li>
                <li><strong>[Second Major Conclusion]</strong> - Brief explanation of your second key finding</li>
                <li><strong>[Third Major Conclusion]</strong> - Brief explanation of your third key finding</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    # Interactive Insight Explorer
    st.subheader("🔍 Explore Specific Insights")

    insight_choice = st.selectbox(
        "Select an insight to learn more:",
        ["Select an insight...",
         "Insight 1: [Brief Title]",
         "Insight 2: [Brief Title]",
         "Insight 3: [Brief Title]"]
    )

    if insight_choice != "Select an insight...":
        st.info(f"""
        **Detailed Explanation:**

        [Provide detailed explanation of the selected insight here. Include:
        - What the data shows
        - Why this is important
        - Statistical significance
        - Business implications]
        """)

    # Actionable Recommendations
    st.subheader("✅ Actionable Recommendations")

    st.markdown("""
        <div class="insight-box">
            <h4>Recommendation 1: [Title]</h4>
            <p><strong>Action:</strong> [Specific, actionable recommendation based on your findings]</p>
            <p><strong>Expected Impact:</strong> [What outcomes can be expected from implementing this]</p>
            <p><strong>Priority:</strong> High/Medium/Low</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="insight-box">
            <h4>Recommendation 2: [Title]</h4>
            <p><strong>Action:</strong> [Second specific recommendation]</p>
            <p><strong>Expected Impact:</strong> [Expected outcomes]</p>
            <p><strong>Priority:</strong> High/Medium/Low</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="insight-box">
            <h4>Recommendation 3: [Title]</h4>
            <p><strong>Action:</strong> [Third specific recommendation]</p>
            <p><strong>Expected Impact:</strong> [Expected outcomes]</p>
            <p><strong>Priority:</strong> High/Medium/Low</p>
        </div>
    """, unsafe_allow_html=True)

    # Limitations and Future Work
    st.subheader("⚠️ Limitations & Future Work")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="warning-box">
                <h4>Limitations</h4>
                <ul>
                    <li>[Limitation 1 - e.g., sample size constraints]</li>
                    <li>[Limitation 2 - e.g., data quality issues]</li>
                    <li>[Limitation 3 - e.g., external validity concerns]</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="info-box">
                <h4>Future Work</h4>
                <ul>
                    <li>[Future direction 1 - e.g., additional variables to explore]</li>
                    <li>[Future direction 2 - e.g., alternative methodologies]</li>
                    <li>[Future direction 3 - e.g., longitudinal analysis]</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Summary Visualization
    st.subheader("📊 Summary Dashboard")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Key Findings", "3", "Major Insights")

    with summary_col2:
        st.metric("Recommendations", "3", "Actionable Steps")

    with summary_col3:
        st.metric("Impact Potential", "High", "Business Value")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 2rem;'>
        <p><strong>Created for:</strong> [Course Name] | <strong>By:</strong> [Your Name] | <strong>Date:</strong> [Presentation Date]</p>
    </div>
""", unsafe_allow_html=True)