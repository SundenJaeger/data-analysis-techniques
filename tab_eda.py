import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def show(df, df_filtered):
    st.title("📊 Exploratory Data Analysis")
    st.markdown("""
    This section dives into the structure of our data, our cleaning process, and the relationships 
    between variables before we apply Machine Learning.
    """)

    dark_params = {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "grid.color": "gray",
        "grid.alpha": 0.3
    }

    # --- SECTION 1: DATA PREPARATION ---
    with st.expander("🛠️ Data Cleaning & Preparation Steps", expanded=False):
        st.markdown("""
        To ensure our K-Means model works correctly, we performed the following cleaning steps:
        1.  **Outlier Handling (CGPA):** We detected unrealistic outliers in CGPA and capped values at **10.0**. 
        2.  **Binary Encoding:** Converted categorical variables (`Placement`, `Internship_Experience`) into binary (0/1) format.
        3.  **Feature Scaling:** Applied `StandardScaler` to normalize ranges.
        """)

    st.divider()

    # --- SECTION 2: UNIVARIATE ANALYSIS ---
    st.subheader("1. Feature Distributions")

    col_dist_graph, col_dist_text = st.columns([2, 1])

    with col_dist_graph:
        feature = st.selectbox("Select Feature to Visualize:",
        ['IQ', 'CGPA', 'Communication_Skills', 'Projects_Completed', 'Academic_Performance'])

        with plt.rc_context(dark_params):
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(df_filtered[feature], kde=True, color='#00FFFF', bins=20, ax=ax)
            ax.set_title(f"Distribution of {feature}", color='white')
            plt.tight_layout() 
            st.pyplot(fig)

    with col_dist_text:
        st.info(f"💡 **Analysis:**")
        if feature == 'IQ':
            st.markdown("IQ follows a **Normal Distribution** (Bell Curve).")
        elif feature == 'CGPA':
            st.markdown("Notice the spike at **10.0** due to outlier capping.")
        elif feature == 'Communication_Skills':
            st.markdown("Slightly left-skewed; students rate themselves highly.")
        else:
            st.markdown("Fairly uniform distribution across the board.")

    st.divider()

    # --- SECTION 3: CORRELATION ANALYSIS ---
    st.subheader("2. Correlation Analysis")

    col_corr_1, col_corr_2 = st.columns([1.5, 1])

    with col_corr_1:
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        with plt.rc_context(dark_params):
            fig_corr, ax_corr = plt.subplots(figsize=(6, 5))

            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                vmin=-1, vmax=1, center=0,
                cbar=False,
                ax=ax_corr,
                annot_kws={"size": 6}
            )

            ax_corr.tick_params(axis='both', which='major', labelsize=7)

            plt.tight_layout()
            st.pyplot(fig_corr)

    with col_corr_2:
        st.write("#### 🏆 Top Predictors")
        st.write("Correlation with `Placement`:")

        target_corr = corr_matrix['Placement_Binary'].sort_values(ascending=False).drop('Placement_Binary')

        st.dataframe(
            target_corr.to_frame().style.background_gradient(cmap='icefire', vmin=-1, vmax=1),
            height=300
        )

    st.divider()

    # --- SECTION 4: MULTIVARIATE ANALYSIS ---
    st.subheader("3. The 'Merit' Hypothesis Check")
    st.markdown("We mapped students into **Four Quadrants** to see if Merit (IQ + Grades) guarantees success.")

    col_scat_1, col_scat_2 = st.columns([2, 1])

    with col_scat_1:
        with plt.rc_context(dark_params):
            fig_scat, ax_scat = plt.subplots(figsize=(7, 4))

            sns.scatterplot(
                data=df_filtered,
                x='IQ',
                y='CGPA',
                hue='Placement',
                style='Placement',
                alpha=0.8,
                palette='seismic',
                ax=ax_scat,
                s=30,
                edgecolor='white',
                linewidth=0.3
            )

            plt.axhline(y=8, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=110, color='gray', linestyle='--', alpha=0.5)

            ax_scat.set_title("IQ vs CGPA Quadrants", color='white')

            legend = ax_scat.legend(title="Placement Status", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.setp(legend.get_title(), color='white')
            for text in legend.get_texts():
                text.set_color("white")
            legend.get_frame().set_facecolor('black')
            legend.get_frame().set_edgecolor('white')

            plt.tight_layout()
            st.pyplot(fig_scat)

    with col_scat_2:
        st.markdown("#### 🔍 Quadrant Analysis")
        st.success("""
        **1. The Safe Zone (Top-Right):**
        * **High IQ (>110) & High CGPA (>8.5)**
        * Consistent placement density.
        """)
        st.info("""
        **2. The Hard Workers (Top-Left & Bottom-Right):**
        * **Top-Left:** Avg IQ but High CGPA.
        * **Bottom-Right:** High IQ but Low CGPA.
        * *Observation:* Leveraging specific strengths.
        """)
        st.error("""
        **3. The Underachievers (Bottom-Left):**
        * **Low IQ (<110) & Low CGPA (<8.5)**
        * *Observation:* Lowest placement chance.
        """)