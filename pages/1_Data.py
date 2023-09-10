import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes


### Seletores: selectbox
dataset_name = st.sidebar.selectbox("Selecionar Dataset: :tulip: :reminder_ribbon: :wine_glass: :syringe:",
                                    ("Iris", "Câncer de Mama", "Vinhos", "Diabetes"))

#Função para carregar os dados wine
@st.cache_data
def load_data(dataset_name): 
    if dataset_name == "Iris":
        data = load_iris()
        df =pd.DataFrame(data.data, columns=data.feature_names)
    elif dataset_name == "Câncer de Mama":
        data = load_breast_cancer()
        df =pd.DataFrame(data.data, columns=data.feature_names)
    elif dataset_name == "Vinhos":
        data = load_wine()
        df =pd.DataFrame(data.data, columns=data.feature_names)
    else:
        data = load_diabetes()  
        df =pd.DataFrame(data.data, columns=data.feature_names)

    return df

dataset = load_data(dataset_name)


#####Layout#####
st.write("### Dataset:game_die:")
st.text(dataset_name)
st.write(dataset, use_container_width=True)
