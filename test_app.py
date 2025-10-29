import streamlit as st

st.title("🚀 BSE MLOps - Test Deployment")
st.write("This is a simple test to verify Streamlit Cloud deployment works.")
st.write("Repository: https://github.com/Wodenvase/BSE-MLOps")
st.write("If you can see this, the deployment is successful!")

if st.button("Test Button"):
    st.success("✅ Streamlit Cloud deployment is working!")
    st.balloons()
