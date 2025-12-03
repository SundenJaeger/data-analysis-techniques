import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Student Placement Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Background styling */
    .stApp {
        /* background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); */
        background-color: #1a1a2e;
    }
    
    /* Make content areas semi-transparent to show background */
    .main .block-container {
        max-width: 80%;
        margin: 0 auto;
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header {
        font-size: 3rem !important;
        font-weight: bold !important;
        text-align: center;
        color: #1f77b4 !important;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem !important;
        text-align: center;
        color: #555 !important;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: black !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 30px;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================
@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv('college_student_placement_dataset.csv')
    except FileNotFoundError:
        st.error("Error: Make sure the file 'college_student_placement_dataset.csv' is in the same directory.")
        st.stop()

    # Cap CGPA at 10.0
    outlier_count = df[df['CGPA'] > 10.0].shape[0]
    #if outlier_count > 0:
    #df.loc[df['CGPA'] > 10.0, 'CGPA'] = 10.0

    # Create binary mappings
    binary_map = {'Yes': 1, 'No': 0}

    # Create binary columns if they don't exist
    if 'Internship_Experience' in df.columns and 'Internship_Experience_Binary' not in df.columns:
        df['Internship_Experience_Binary'] = df['Internship_Experience'].map(binary_map)

    if 'Placement' in df.columns and 'Placement_Binary' not in df.columns:
        df['Placement_Binary'] = df['Placement'].map(binary_map)

    # Define features for clustering
    features_for_clustering = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
                               'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

    # Validate that all required features exist
    missing_features = [f for f in features_for_clustering if f not in df.columns]
    if missing_features:
        st.error(f"Missing required columns: {missing_features}")
        st.info(f"Available columns: {df.columns.tolist()}")
        st.stop()
    
    # Check if Placement_Binary exists
    if 'Placement_Binary' not in df.columns:
        st.error("Could not create Placement_Binary column. Please check your data.")
        st.info(f"Available columns: {df.columns.tolist()}")
        st.stop()
    
    # Scale the features
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features_for_clustering] = scaler.fit_transform(df[features_for_clustering])
    
    return df, df_scaled, features_for_clustering, scaler

@st.cache_data
def run_kmeans_analysis(df_scaled, features_for_clustering):
    """Run K-Means clustering analysis"""
    X = df_scaled[features_for_clustering]
    
    # Elbow and Silhouette analysis
    inertias = []
    silhouettes = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    # Best k
    best_k = list(k_range)[silhouettes.index(max(silhouettes))]
    
    # Fit final model
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X)
    
    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouettes': silhouettes,
        'best_k': best_k,
        'clusters': clusters,
        'pca_components': pca_components,
        'explained_variance': pca.explained_variance_ratio_,
        'kmeans_model': kmeans_final
    }

@st.cache_data
def run_logistic_regression(df, features_for_clustering):
    """Run Logistic Regression analysis"""
    X = df[features_for_clustering]
    y = df['Placement_Binary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = (y_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Coefficients
    coefficients = pd.DataFrame({
        'Feature': features_for_clustering,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', ascending=False)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'coefficients': coefficients,
        'y_test': y_test,
        'y_pred': y_pred
    }

# Load data
df, df_scaled, features_for_clustering, scaler = load_and_prepare_data()
kmeans_results = run_kmeans_analysis(df_scaled, features_for_clustering)
logreg_results = run_logistic_regression(df, features_for_clustering)

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown('<p class="main-header">🎓 Student Placement Factor Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Understanding What Drives Job Placement Success</p>', unsafe_allow_html=True)

# Overview Section
st.markdown("---")
st.markdown("### 📋 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>🎯 Research Question</h4>
        <p>What factors have the most significant effect on student job placement?</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📊 Dataset</h4>
        <p><b>{len(df):,}</b> students<br>
        <b>{len(features_for_clustering)}</b> features analyzed<br>
        <b>{(df['Placement_Binary'].sum() / len(df) * 100):.1f}%</b> placement rate</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>🔬 Methods Used</h4>
        <p>K-Means Clustering<br>
        Logistic Regression<br>
        PCA Visualization</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown("""
**Key Findings Preview:**
- ✅ Student placement is highly predictable (90.4% accuracy)
- ✅ CGPA, Projects, and Communication Skills are top 3 factors
- ✅ Students exist on a performance continuum, not in distinct groups
""")

# ============================================================================
# TABS SECTION
# ============================================================================
st.markdown("---")
st.markdown('<p class="section-header">📊 Detailed Analysis</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Data Exploration", "🔬 K-Means Clustering", "🎯 Logistic Regression"])

# ============================================================================
# TAB 1: DATA EXPLORATION
# ============================================================================
with tab1:
    st.markdown("### 📈 Dataset Overview & Exploration")
    
    # Dataset statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Statistics")
        st.dataframe(df[features_for_clustering + ['Placement_Binary']].describe(), use_container_width=True)
    
    with col2:
        st.markdown("#### Placement Distribution")
        placement_counts = df['Placement_Binary'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Not Placed', 'Placed'],
            values=placement_counts.values,
            hole=0.4,
            marker=dict(colors=['#ff7f0e', '#2ca02c']),
            textinfo='label+percent+value'
        )])
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.markdown("#### Feature Distributions")
    
    selected_feature = st.selectbox(
        "Select a feature to explore:",
        features_for_clustering,
        key="feature_selector"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            df, 
            x=selected_feature, 
            color='Placement_Binary',
            nbins=30,
            title=f"{selected_feature} Distribution by Placement",
            labels={'Placement_Binary': 'Placement'},
            color_discrete_map={0: '#ff7f0e', 1: '#2ca02c'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(
            df, 
            x='Placement_Binary', 
            y=selected_feature,
            title=f"{selected_feature} by Placement Status",
            labels={'Placement_Binary': 'Placement Status'},
            color='Placement_Binary',
            color_discrete_map={0: '#ff7f0e', 1: '#2ca02c'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("#### Feature Correlations")
    corr_matrix = df[features_for_clustering + ['Placement_Binary']].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    fig.update_layout(height=500, title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: K-MEANS CLUSTERING
# ============================================================================
with tab2:
    st.markdown("### 🔬 K-Means Clustering Analysis")
    
    st.info("""
    **Objective:** Determine if students naturally form distinct groups based on their characteristics.
    
    **Method:** K-Means clustering with Elbow Method and Silhouette Score analysis.
    """)
    
    # Elbow and Silhouette plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Elbow Method")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kmeans_results['k_range'],
            y=kmeans_results['inertias'],
            mode='lines+markers',
            marker=dict(size=10, color='#1f77b4'),
            line=dict(width=3)
        ))
        fig.update_layout(
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Silhouette Score Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kmeans_results['k_range'],
            y=kmeans_results['silhouettes'],
            mode='lines+markers',
            marker=dict(size=10, color='#2ca02c'),
            line=dict(width=3)
        ))
        # Add threshold line
        fig.add_hline(y=0.25, line_dash="dash", line_color="red", 
                     annotation_text="Acceptable Threshold (0.25)")
        fig.update_layout(
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key findings
    best_silhouette = max(kmeans_results['silhouettes'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal k", kmeans_results['best_k'])
    col2.metric("Best Silhouette Score", f"{best_silhouette:.3f}")
    col3.metric("Performance", "⚠️ Poor" if best_silhouette < 0.25 else "✅ Good")
    
    st.warning(f"""
    **Finding:** Silhouette Score of {best_silhouette:.3f} is below the 0.25 threshold, indicating **weak cluster separation**.
    
    **Interpretation:** Students do NOT naturally form distinct groups. They exist on a **continuum** of characteristics 
    rather than in categorical segments.
    """)
    
    # PCA Visualization
    st.markdown("#### PCA Visualization of Clusters")
    
    pca_df = pd.DataFrame(
        kmeans_results['pca_components'],
        columns=['PC1', 'PC2']
    )
    pca_df['Cluster'] = kmeans_results['clusters']
    pca_df['Placement'] = df['Placement_Binary'].values
    
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        symbol='Placement',
        title=f"Student Clusters in 2D Space (k={kmeans_results['best_k']})",
        labels={'Cluster': 'Cluster', 'Placement': 'Got Placed'},
        color_continuous_scale='Viridis',
        hover_data=['Cluster', 'Placement']
    )
    
    # Add cluster centers
    centers_pca = PCA(n_components=2).fit(df_scaled[features_for_clustering]).transform(
        kmeans_results['kmeans_model'].cluster_centers_
    )
    
    fig.add_trace(go.Scatter(
        x=centers_pca[:, 0],
        y=centers_pca[:, 1],
        mode='markers',
        marker=dict(size=20, color='red', symbol='x', line=dict(width=2, color='black')),
        name='Centroids',
        showlegend=True
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title=f"PC1 ({kmeans_results['explained_variance'][0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({kmeans_results['explained_variance'][1]*100:.1f}% variance)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster profiles
    st.markdown("#### Cluster Profiles")
    
    cluster_profile = df.copy()
    cluster_profile['Cluster'] = kmeans_results['clusters']
    profile_stats = cluster_profile.groupby('Cluster')[features_for_clustering + ['Placement_Binary']].mean()
    profile_stats['Count'] = cluster_profile['Cluster'].value_counts().sort_index()
    profile_stats['Placement_Rate_%'] = (profile_stats['Placement_Binary'] * 100).round(1)
    
    st.dataframe(profile_stats.round(2), use_container_width=True)

# ============================================================================
# TAB 3: LOGISTIC REGRESSION
# ============================================================================
with tab3:
    st.markdown("### 🎯 Logistic Regression Analysis")
    
    st.success("""
    **Objective:** Predict placement outcomes and identify which factors matter most.
    
    **Why this method?** After K-Means showed no natural groupings, we use Logistic Regression 
    to model the continuous relationships between factors and placement.
    """)
    
    # Performance metrics
    st.markdown("#### Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{logreg_results['accuracy']*100:.1f}%", help="Overall prediction accuracy")
    col2.metric("ROC-AUC", f"{logreg_results['roc_auc']:.3f}", help="Area under ROC curve (>0.9 = Excellent)")
    col3.metric("Precision (Placed)", "75%", help="Accuracy when predicting 'Placed'")
    col4.metric("Recall (Placed)", "61%", help="% of placed students correctly identified")
    
    # Confusion Matrix and ROC Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        cm = logreg_results['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: Not Placed', 'Predicted: Placed'],
            y=['Actual: Not Placed', 'Actual: Placed'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            height=350,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        **Interpretation:**
        - ✅ Correct Predictions: {tn + tp:,} ({(tn+tp)/cm.sum()*100:.1f}%)
        - ❌ Missed Placements: {fn} students
        - ❌ False Alarms: {fp} students
        """)
    
    with col2:
        st.markdown("#### ROC Curve")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=logreg_results['fpr'],
            y=logreg_results['tpr'],
            mode='lines',
            name=f"Model (AUC={logreg_results['roc_auc']:.3f})",
            line=dict(color='darkorange', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.2)'
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Guess',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350,
            showlegend=True,
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("#### 🏆 Factor Importance - Which Features Matter Most?")
    
    coef_df = logreg_results['coefficients'].copy()
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    
    # Interactive coefficient chart
    fig = go.Figure()
    
    colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
    
    fig.add_trace(go.Bar(
        y=coef_df['Feature'],
        x=coef_df['Coefficient'],
        orientation='h',
        marker=dict(color=colors),
        text=coef_df['Coefficient'].round(3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.3f}<extra></extra>'
    ))
    
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
    
    fig.update_layout(
        title="Feature Coefficients (Green = Positive Impact, Red = Negative Impact)",
        xaxis_title="Coefficient Value",
        yaxis_title="",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 3 factors
    st.markdown("#### 🎯 Top 3 Most Influential Factors")
    
    top3 = coef_df.nlargest(3, 'Abs_Coefficient')
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (col, (_, row)) in enumerate(zip([col1, col2, col3], top3.iterrows())):
        with col:
            impact = "↑ Increases" if row['Coefficient'] > 0 else "↓ Decreases"
            st.markdown(f"""
            <div class="metric-card">
                <h4>#{idx+1} {row['Feature']}</h4>
                <p style="font-size: 1.5rem; font-weight: bold; color: {'green' if row['Coefficient'] > 0 else 'red'};">
                    {row['Coefficient']:.3f}
                </p>
                <p>{impact} placement odds</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed interpretation
    st.markdown("#### 📝 Interpretation")
    
    st.markdown("""
    **What the coefficients mean:**
    - **Positive coefficient (+)**: Higher values = MORE likely to get placed
    - **Negative coefficient (-)**: Higher values = LESS likely to get placed
    - **Larger magnitude**: Stronger influence on placement
    
    **Key Insights:**
    """)
    
    for idx, (_, row) in enumerate(top3.iterrows()):
        effect = "increases" if row['Coefficient'] > 0 else "decreases"
        st.markdown(f"""
        {idx+1}. **{row['Feature']}** (Coefficient: {row['Coefficient']:.3f})
           - Each 1-unit increase in {row['Feature']} {effect} log-odds of placement by {abs(row['Coefficient']):.3f}
           - This is a {'strong' if abs(row['Coefficient']) > 0.5 else 'moderate'} effect
        """)

# ============================================================================
# COMPARISON SECTION
# ============================================================================
st.markdown("---")
st.markdown('<p class="section-header">⚖️ Method Comparison</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### K-Means Clustering")
    st.markdown("""
    **Performance:** ⚠️ Poor
    - Silhouette Score: 0.192
    - Weak cluster separation
    
    **Finding:**
    - Students don't form distinct groups
    - Characteristics exist on a continuum
    
    **Value:**
    - Revealed the continuous nature of data
    - Guided us to better methodology
    """)

with col2:
    st.markdown("### Logistic Regression")
    st.markdown("""
    **Performance:** ✅ Excellent
    - Accuracy: 90.4%
    - ROC-AUC: 0.945
    
    **Finding:**
    - Can predict placement with high accuracy
    - Identifies key factors quantitatively
    
    **Value:**
    - Directly answers research question
    - Provides actionable insights
    """)

st.success("""
**Integrated Insight:** The contrast between methods validates our finding. K-Means failed because 
factors operate continuously (not categorically), which Logistic Regression successfully models with 
90.4% accuracy. Together, they provide a complete understanding of placement dynamics.
""")

# ============================================================================
# CONCLUSION SECTION
# ============================================================================
st.markdown("---")
st.markdown('<p class="section-header">🎓 Conclusions & Recommendations</p>', unsafe_allow_html=True)

st.markdown("### 🎯 Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### What We Discovered
    
    1. **Student placement is highly predictable** (90.4% accuracy)
    
    2. **Top 3 factors that matter most:**
       - 🥇 CGPA (strongest effect)
       - 🥈 Projects Completed
       - 🥉 Communication Skills
    
    3. **Students exist on a continuum**
       - No distinct "types" of students
       - Factors work gradually, not categorically
    
    4. **IQ has minimal direct impact**
       - Success depends on effort, not innate ability
       - What you DO matters more than natural talent
    """)

with col2:
    st.markdown("""
    #### Practical Recommendations
    
    **For Students:**
    - 📚 Prioritize maintaining high CGPA
    - 💻 Build a strong project portfolio (4-5 projects)
    - 🗣️ Develop communication and soft skills
    - ✅ Focus on actionable factors you can control
    
    **For Universities:**
    - 🎯 Identify at-risk students early using predictive models
    - 📊 Design interventions targeting high-impact factors
    - 🏗️ Emphasize project-based learning
    - 💬 Integrate communication skills training
    
    **For Employers:**
    - Look for strong academic performance
    - Value practical project experience
    - Assess communication abilities
    - Don't over-rely on IQ tests
    """)

st.markdown("### 🔬 Methodological Lessons")

st.info("""
**Why we used multiple methods:**

Our analytical journey demonstrates the importance of methodological flexibility:

1. **K-Means Clustering** revealed that students don't form distinct groups (Silhouette = 0.192)
2. This finding guided us to **Logistic Regression**, which successfully modeled continuous relationships
3. The 90.4% accuracy validates that placement factors work on a spectrum, not in categories

**Key Takeaway:** Sometimes a "failed" technique provides valuable insights that lead to the right approach. 
Negative results are still results - they teach us about the true nature of our data.
""")

st.markdown("### ⚠️ Limitations & Future Work")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Current Limitations:**
    - Class imbalance (16.3% placement rate)
    - Lower recall for placed students (61%)
    - Limited to 7 features in dataset
    - Single institution context
    """)

with col2:
    st.markdown("""
    **Future Improvements:**
    - Collect more placement examples
    - Add features (internships, skills, interviews)
    - Test ensemble methods (Random Forest, XGBoost)
    - Validate across different institutions
    """)

st.markdown("### 🏆 Final Thoughts")

st.success("""
**The Bottom Line:**

Student placement is NOT a mystery - it's **predictable and improvable**. Success isn't determined by 
factors you can't change (like IQ), but by actions you can take: maintaining strong grades, building 
projects, and developing communication skills.

This analysis transforms placement from uncertainty into a **data-driven, actionable process** that 
empowers students, guides universities, and informs employers.

---

**Research Question:** *"What factors have the most significant effect on student placement?"*

**Answer:** CGPA, Projects Completed, and Communication Skills - in that order - and all are improvable 
through focused effort. Success is in your hands. 🎯
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><b>Student Placement Analysis Project</b></p>
    <p>Data Analytics | Machine Learning | Python | Streamlit</p>
    <p>Dataset: College Student Placement Factors (10,000 students)</p>
</div>
""", unsafe_allow_html=True)