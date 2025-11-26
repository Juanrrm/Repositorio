import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Despliegue Autogluon - NeuralNetFastAI", layout="centered")
st.title("Despliegue: NeuralNetFastAI_r88_BAG_L1")
st.markdown("Carga el archivo `model.pkl` (colócalo en la misma carpeta que esta app) y completa cada variable. Pulsa 'Predecir' para obtener la predicción.")

FEATURES = [
    "EDAD",
    "PUNTAJE TEST",
    "Profundidad G.Cronica",
    "Grado G.cronica",
    "Act. Inflamatoria",
    "Grado Act. Inflamatoria",
    "Daño mucinoso",
    "Grado de Daño mucinosos",
    "Extension de Daño Mucinoso",
    "Numero de Foliculos linfoides",
]

# Sidebar: permite subir model.pkl o usar el ya presente en el directorio
st.sidebar.header("Modelo")
uploaded = st.sidebar.file_uploader("Sube model.pkl (opcional)", type=["pkl"])
model_path = None
if uploaded is not None:
    model_path = os.path.join("/tmp", "uploaded_model.pkl")
    with open(model_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success("Modelo subido a /tmp/uploaded_model.pkl")
else:
    # intenta usar model.pkl en el mismo directorio
    default_path = "model.pkl"
    if os.path.exists(default_path):
        model_path = default_path
        st.sidebar.info(f"Usando {default_path} en el directorio actual")
    else:
        st.sidebar.warning("No se ha encontrado model.pkl en el directorio. Puedes subirlo desde aquí.")

# Inputs: un cuadro de texto por variable
st.header("Entradas del usuario")
inputs = {}
cols = st.columns(2)
for i, feat in enumerate(FEATURES):
    # alterna columnas para aspecto más compacto
    with cols[i % 2]:
        val = st.text_input(feat, value="", key=feat)
        inputs[feat] = val

st.write("---")

# Botón de predicción
if st.button("Predecir"):
    if model_path is None:
        st.error("No hay ningún modelo cargado. Sube model.pkl o ponlo en el mismo directorio de la app.")
    else:
        # Construir DataFrame de una fila con las entradas
        df = pd.DataFrame([inputs])

        # Intentar convertir columnas a numéricas (si aplicable)
        for c in df.columns:
            # intenta convertir a float, si falla, mantén la cadena (modelos tabulares suelen necesitar numéricos)
            try:
                df[c] = pd.to_numeric(df[c].str.replace(',', '.'))
            except Exception:
                # si la columna ya es numérica o no se puede convertir, dejamos el valor tal cual
                try:
                    df[c] = pd.to_numeric(df[c])
                except Exception:
                    pass

        # Cargar el modelo con pickle
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.exception(e)
            st.error("Error cargando model.pkl. Asegúrate de que el archivo corresponde a un predictor de AutoGluon o un objeto con método `predict`.")
        else:
            st.subheader("Data para predecir")
            st.dataframe(df)

            # Intentar predecir con diferentes llamadas según tipo de objeto
            try:
                # AutoGluon TabularPredictor: tiene método .predict_proba y .predict
                if hasattr(model, "predict_proba") and hasattr(model, "predict"):
                    y_pred = model.predict(df)
                    try:
                        y_proba = model.predict_proba(df)
                    except Exception:
                        y_proba = None

                    st.success("Predicción realizada con éxito (AutoGluon compatible detectado)")
                    st.write("**Predicción (label):**")
                    st.write(y_pred.tolist())
                    if y_proba is not None:
                        st.write("**Probabilidades (si aplica):**")
                        st.dataframe(pd.DataFrame(y_proba, columns=[f"p_{c}" for c in range(y_proba.shape[1])]))
                else:
                    # Genérico: si tiene predict
                    if hasattr(model, "predict"):
                        y_pred = model.predict(df)
                        st.success("Predicción realizada con éxito")
                        st.write(y_pred)
                    else:
                        st.error("El objeto cargado no tiene un método `predict` ni `predict_proba`. No se puede usar para inferencia directa.")
            except Exception as e:
                st.exception(e)
                st.error("Error durante la predicción. Revisa que las columnas y tipos coincidan con los que el modelo espera.")

st.write("---")
st.markdown("**Notas:**\n - Asegúrate de que los nombres de columnas (features) coincidan exactamente con los que usó el modelo durante el entrenamiento.\n- Si tu modelo necesita preprocesamiento (one-hot encoding, scaling, etc.), y esos pasos no están embebidos en `model.pkl`, tendrás que replicarlos antes de pasar el DataFrame al predictor.\n- Para ejecutar la app: `streamlit run streamlit_autogluon_app.py` en la carpeta donde están `streamlit_autogluon_app.py` y `model.pkl`.")