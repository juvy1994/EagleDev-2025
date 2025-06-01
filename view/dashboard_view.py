import streamlit as st
import os
from controller.predictor_controller import PredictorController

# Ruta del modelo
MODEL_PATH = "scripts/covid_diagnosis_model.tflite"
TEMP_IMAGE_DIR = "images/temp_uploads"

def save_uploaded_file(uploaded_file):
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
    file_path = os.path.join(TEMP_IMAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.set_page_config(page_title="COVID-19 Diagnosis", layout="centered")
    st.title("🩺 Diagnóstico de COVID-19 con Rayos X")
    st.write("Sube una radiografía para detectar si hay signos de **COVID-19**, **neumonía viral** o si los **pulmones están normales**.")

    uploaded_file = st.file_uploader("📤 Subir imagen", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image_path = save_uploaded_file(uploaded_file)

        st.image(image_path, caption="📷 Imagen subida", use_container_width=True)

        # Llamar al controlador (modelo)
        predictor = PredictorController(model_path=MODEL_PATH)

        with st.spinner("🔍 Analizando imagen..."):
            try:
                diagnosis, confidence = predictor.predict_image(image_path)
                st.success(f"✅ Diagnóstico: **{diagnosis}**\n\n🧠 Confianza del modelo: `{confidence:.2f}`")
            except Exception as e:
                st.error(f"❌ Error al procesar la imagen: {str(e)}")

if __name__ == "__main__":
    main()
