## Algoritmo não supervisionado de Machine Learning, classificadores: KNN, SVC, Random Forest
## Datasets: IRIS, Breast_Cancer, Wine, diabetes.
## Selecionar Dataset e tipo de classsificador de acordo com cada parametro. E saída Acurácia do modelo e o gráfico Plot.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from PIL import Image


### Layout Streamlit
st.header("Análise de Classificadores de Machine Learning não supervisionado :bar_chart: :chart_with_downwards_trend:")

st.write("#### Explorar diferentes classificadores. Qual é o melhor?")
st.divider()

### Seletores: selectbox
dataset_name = st.sidebar.selectbox("Selecionar Dataset: :tulip: :reminder_ribbon: :wine_glass: :syringe:",
                                    ("Iris", "Câncer de Mama", "Vinhos", "Diabetes"))

classifier_names = st.sidebar.selectbox("Selecionar Classificador: :calendar: :books: ",
                                    ("KNN", "SVM", "Random Forest", "Decision Tree"))         
                          
st.write(f"Classificador: :blue[{classifier_names}]")

### Carregar os datasets utilizando funções.
@st.cache_data
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Câncer de Mama":
        data = load_breast_cancer()
    elif dataset_name == "Vinhos":
        data = load_wine()
    else:
        data = load_diabetes()  
           
    X = data.data
    y = data.target
   
    return X, y

X, y = get_dataset(dataset_name)

st.write(f"Tamanho do Dataset: :red[{X.shape}] ")
st.write(f"Número de Classes: :blue[{len(np.unique(y))}] ")

##Função selecionar os parametros dos algoritmos classificadores.
def add_parameter_ui(clf_names):
    params = dict()
    if clf_names == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_names == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_names == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 20)
        n_estimators = st.sidebar.slider("n_estimators", 10, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    else:
        max_depth = st.sidebar.slider("max_depth", 1, 10)
        max_leaf_nodes = st.sidebar.slider("max_leaf_nodes", 1, 10)
        params["max_depth"] = max_depth
        params["max_leaf_nodes"] = max_leaf_nodes
    
    return params

params = add_parameter_ui(classifier_names)       

##Função para selecionar os classificadores de acordo com os parametros selecionados.
def get_classifier(clf_names, params):
    if clf_names == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_names == "SVM":
        clf = SVC(C=params["C"])
    elif clf_names == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    else:
        clf = DecisionTreeClassifier(max_depth=params["max_depth"], max_leaf_nodes=params["max_leaf_nodes"], random_state=1234)     
    return clf

clf = get_classifier(classifier_names, params)     


##Divisão dos dados (Treino e Teste).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

##Treinamento do modelo: (X_train, y_train)
clf.fit(X_train, y_train)

## Previsão do modelo: X_test
y_pred = clf.predict(X_test)

## Métricas de classificação do modelo entre (y_test) e (y_pred): teste x predição
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

st.write(f"Acurácia = :red[{acc:.2f}]")
st.write(f"Precisão = :green[{precision:.2f}]")
st.write(f"Recall = :blue[{recall:.2f}]")
st.write(f"F1 = :violet[{f1:.2f}]")


#Tab
tab1 , tab2 = st.tabs(["Plot", "Chart-Bar"])

## Plotar o gráfico - 2D
with tab1:
    st.write("#### Gráfico Plot")
    pca = PCA(2)
    x_projected = pca.fit_transform(X)

    x1 = x_projected[:, 0]
    x2 = x_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar()
    #plt.show()
    st.pyplot(fig)

     
# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)  

st.sidebar.divider()
st.sidebar.image("logo.png", width=200)

