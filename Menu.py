import streamlit as st

st.set_page_config(
    page_title="Educationnal TS-LDDMM app",
    page_icon="👋",
)

st.write("# Welcome ! 👋")
st.markdown("""
To understand the effect of the hyperparameters on a geodesic shooting with the 
TS-LDDMM kernel you can use the app "parameter_shooting".
 The graph of reference is fixed and the initial velocity field is sampled randomly.

To understand the effect of the hyperparameters on a Registration 
you can use the app "parameter_shooting". 
A graph of reference and a target are fixed and we try to recover a velocity field to map the graph of reference to the target with a TS-LDDMM shooting.
""")

st.sidebar.success("Select a demo above.")


# streamlit run Menu.py


