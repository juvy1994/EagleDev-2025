import numpy as np
import tensorflow as tf  # ✅ Usamos TensorFlow en lugar de tflite_runtime
import cv2
from model.class_mapping import class_mapping  # Diccionario de clases

class TFLiteModel:
    def __init__(self, model_path):
        """
        Inicializa el intérprete TFLite usando TensorFlow (compatible con Windows).
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, image_path):
        """
        Carga la imagen, la redimensiona a 150x150 y la normaliza.
        Convierte a formato compatible para MobileNetV2 (1 canal, batch 1).
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
        
        img = cv2.resize(img, (150, 150))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # (150, 150, 1)
        img = np.expand_dims(img, axis=0)   # (1, 150, 150, 1)
        return img

    def predict(self, image_path):
        """
        Ejecuta inferencia sobre una imagen y devuelve la clase predicha y su confianza.
        """
        input_data = self.preprocess(image_path)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Softmax si la suma no da 1
        if not np.allclose(np.sum(output), 1, atol=1e-3):
            exp_out = np.exp(output - np.max(output))
            probabilities = exp_out / np.sum(exp_out)
        else:
            probabilities = output

        class_index = np.argmax(probabilities)
        class_name = class_mapping[class_index]
        confidence = probabilities[class_index]
        return class_name, confidence
