from model.tflite_model import TFLiteModel

class PredictorController:
    def __init__(self, model_path):
        self.model = TFLiteModel(model_path)

    def predict_image(self, image_path):
        # La función predict ya preprocesa y devuelve clase y confianza
        predicted_class, confidence = self.model.predict(image_path)
        return predicted_class, confidence
