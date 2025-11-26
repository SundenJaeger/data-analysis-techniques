import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def show(df):
    st.title("🧠 Cluster Analysis (Student Personas)")
    st.markdown("""
    To solve the "overlap" problem seen in the scatter plots, we used **K-Means Clustering** to segment students based on 5 key dimensions:
    *IQ, CGPA, Communication Skills, Projects, and Academic Performance.*
    """)

    # 1. Prepare Data for Model
    features = ['IQ', 'CGPA', 'Communication_Skills', 'Projects_Completed', 'Academic_Performance']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # 2. Run K-Means (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 3. Cluster Profiling
    st.subheader("Identified Student Personas")
    
    # Create the Summary Table
    profile = df.groupby('Cluster')[features + ['Placement_Binary']].mean()
    profile['Placement_Rate_%'] = df.groupby('Cluster')['Placement_Binary'].mean() * 100
    profile['Count'] = df['Cluster'].value_counts()
    
    # Display Table with Highlighting
    st.dataframe(profile.style.highlight_max(axis=0, color='lightgreen').format("{:.2f}"))
    
    st.markdown("""
    **🔍 Key Findings:**
    * **Cluster 0 (The Communicators):** Average Grades but **High Communication**. Highest Placement Rate (~32%).
    * **Cluster 1 (The Silent Grinders):** Highest Grades & Projects but **Low Communication**. Low Placement (~11%).
    * **Cluster 2 (The At-Risk):** Low Projects & Low Communication. Near Zero Placement.
    """)

    st.markdown("---")
    st.subheader("The 'Secret Sauce': Why Cluster 0 Wins")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar Chart: Communication
        fig4, ax4 = plt.subplots()
        sns.barplot(x=profile.index, y=profile['Communication_Skills'], palette='viridis', ax=ax4)
        ax4.set_title("Avg Communication Score by Cluster")
        ax4.set_ylabel("Score (1-10)")
        st.pyplot(fig4)
        st.caption("Observation: Cluster 0 dominates in Communication.")
        
    with col2:
        # Bar Chart: Projects
        fig5, ax5 = plt.subplots()
        sns.barplot(x=profile.index, y=profile['Projects_Completed'], palette='viridis', ax=ax5)
        ax5.set_title("Avg Projects Completed by Cluster")
        ax5.set_ylabel("Count")
        st.pyplot(fig5)
        st.caption("Observation: Cluster 1 works harder (more projects), but fails without communication.")