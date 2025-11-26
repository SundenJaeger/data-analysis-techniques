import streamlit as st

def show(df):
    # Header Section
    st.title("🎓 College Student Placement Analysis")
    st.markdown("### *Data-Driven Insights into Career Success*")
    
    # Team Introduction
    with st.expander("ℹ️ About Team Algorhythym", expanded=True):
        st.markdown("""
        **Developed by:**
        * **Penaflor, Amiel Joshua**
        * **Sitoy, Rene John**
        * **Arbiol, John Lyster**
        * **Perez, John Clyde**
        * **Singson, Sol Angelo**
        """)

    st.markdown("---")

    # Project Description
    col_desc, col_stats = st.columns([2, 1])

    with col_desc:
        st.subheader("📌 Project Goal")
        st.markdown("""
        In the competitive landscape of higher education, students often wonder: *Does a high IQ guarantee a job? Do grades matter more than projects?*
        
        Our goal is to move beyond simple assumptions. Using **Unsupervised Machine Learning (K-Means Clustering)**, we aim to:
        1.  **Dissect** the complex relationship between academic metrics (CGPA, IQ) and practical skills (Communication, Projects).
        2.  **Identify** distinct "Student Personas" that exist within the student body.
        3.  **Reveal** the hidden factors that truly drive placement success.
        """)

    with col_stats:
        st.subheader("📊 Dataset at a Glance")
        st.metric("Total Records", f"{len(df):,}")
        
        # Calculate Placement Rate safely
        placement_rate = df['Placement_Binary'].mean()
        st.metric("Placement Rate", f"{placement_rate:.1%}") 
        
        st.metric("Avg CGPA", f"{df['CGPA'].mean():.2f}")

    st.warning("⚠️ **Dataset Note:** The low placement rate observed (approx 16%) highlights the difficulty of the current job market.")

    with st.expander("🔍 View Raw Dataset (First 10 Rows)"):
        st.dataframe(df.head(10))