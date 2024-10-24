import cv2
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs
import os
import base64

configs = BaseModelConfigs.load(r"Models\handwriting_recognition\202422101411\configs.yaml")

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):

        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

# Initialize the ONNX model
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

def preprocess_and_predict(image_path):
    """Preprocess the image and predict the text."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' not found.")

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image from '{image_path}'.")

    # Predict the text from the image
    prediction_text = model.predict(image)
    _, buffer = cv2.imencode('.png', image) 
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return prediction_text,image_base64