import pandas as pd
import streamlit as st

@st.cache_data
def load_and_clean_data():
    # Load Data
    try:
        df = pd.read_csv('college_student_placement_dataset.csv')
    except FileNotFoundError:
        st.error("csv file not found. Please ensure 'college_student_placement_dataset.csv' is in the same directory.")
        return None

    # Cleaning: Cap CGPA at 10.0 (Handling Outliers)
    if 'CGPA' in df.columns:
        df['CGPA'] = df['CGPA'].clip(upper=10.0)
    
    # Cleaning: ROBUST Mapping for Placement
    if 'Placement' in df.columns:
        df['Placement_Binary'] = df['Placement'].astype(str).str.lower().str.strip().apply(
            lambda x: 1 if x in ['placed', 'yes', 'true', '1'] else 0
        )
    elif 'Placement_Binary' in df.columns:
        df['Placement_Binary'] = df['Placement_Binary'].astype(int)
        
    # Ensure Internship is numeric
    if 'Internship_Experience' in df.columns:
         df['Internship_Experience_Binary'] = df['Internship_Experience'].astype(str).str.lower().str.strip().apply(
            lambda x: 1 if x in ['yes', 'true', '1'] else 0
        )
        
    return df