import streamlit as st
from streamlit_option_menu import option_menu

# Import our custom modules
import data_loader
import tab_overview
import tab_eda
import tab_analysis
import tab_conclusions

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Team Algorhythym | Placement Analysis",
    page_icon="🎓",
    layout="wide"
)

# Load Data using the separate file
df = data_loader.load_and_clean_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1256/1256650.png", width=80) 
    st.title("Team Algorhythym")
    
    selected = option_menu(
        menu_title=None,
        options=["Overview", "Data Exploration", "Analysis & Insights", "Conclusions"],
        icons=["house", "bar-chart", "lightbulb", "check-circle"],
        default_index=0,
    )
    
    st.markdown("---")
    st.markdown("**Filters (Global)**")
    
    # Interactive Slider (Applies to EDA mostly)
    if df is not None:
        min_iq, max_iq = int(df['IQ'].min()), int(df['IQ'].max())
        iq_range = st.slider("Select IQ Range:", min_iq, max_iq, (min_iq, max_iq))
        
        # Create filtered dataframe
        df_filtered = df[(df['IQ'] >= iq_range[0]) & (df['IQ'] <= iq_range[1])]
    else:
        st.stop()

# --- PAGE ROUTING ---
if selected == "Overview":
    tab_overview.show(df)

elif selected == "Data Exploration":
    tab_eda.show(df, df_filtered)

elif selected == "Analysis & Insights":
    tab_analysis.show(df)

elif selected == "Conclusions":
    tab_conclusions.show()