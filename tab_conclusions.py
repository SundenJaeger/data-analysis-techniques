import streamlit as st

def show():
    st.title("💡 Conclusions & Recommendations")
    
    st.success("Final Verdict: Soft Skills are the Tie-Breaker.")
    
    st.markdown("""
    ### Key Takeaways:
    1.  **Grades are Entry Tickets, Not Guarantees:** High CGPA is common among placed students, but High CGPA *without* soft skills leads to failure (as seen in Cluster 1).
    2.  **Communication is the Differentiator:** The cluster with the highest placement rate was NOT the one with the highest IQ, but the one with the highest Communication Skills.
    3.  **The "Silent Grinder" Trap:** Students who focus solely on projects and grades but ignore communication are at high risk of not being placed.

    ### Actionable Advice for Students:
    * **For High Achievers:** Don't neglect soft skills. Join debate clubs or presentation workshops.
    * **For Average Students:** You can outperform "smarter" peers by mastering communication.
    """)