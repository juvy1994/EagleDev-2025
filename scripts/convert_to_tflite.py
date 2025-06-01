import tensorflow as tf
import os

try:
    # Cargar el modelo entrenado
    model = tf.keras.models.load_model('covid_diagnosis_model.h5')
    print("Modelo cargado correctamente.")

    # Convertir el modelo a TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("Modelo convertido a TFLite.")

    # Guardar el modelo convertido
    tflite_model_path = 'covid_diagnosis_model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Modelo guardado como {tflite_model_path}")

    # Verificar que el archivo fue creado
    if os.path.exists(tflite_model_path):
        print("Archivo .tflite creado correctamente.")
    else:
        print("Error: El archivo .tflite NO fue creado.")

except Exception as e:
    print("Error durante la conversión:", e)
