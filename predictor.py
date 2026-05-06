import numpy as np
from PIL import Image
import tensorflow as tf

model_path=r"C:\Users\sarvesh\OneDrive\文档\Desktop\Banana_Life_AI\model\banana_model.keras"
class_path= r"C:\Users\sarvesh\OneDrive\文档\Desktop\Banana_Life_AI\model\class_names.txt"

def load_model():
    return tf.keras.models.load_model(model_path)

def load_class_names():
    with open(class_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_life_info(stage):
    stage= stage.lower()
    
    info= {
        "unripe": {
            "days_left": "5-7 days",
            "freshness_score": 95,
            "advice": "keep at room temperature. It will ripe naturally."
            
        },
        "ripe": {
            "days_left": "2-4 days",
            "freshness_score": 80,
            "advice": "Best time to eat. Store in a cool place"
            
        },
        "overripe": {
            "days_left": "1-2 days",
            "freshness_score": 45,
            "advice": "Use soon. Good for smoothies or banana bread."
        },
        "rotten": {
            "days_left": "0 days",
            "freshness_score": 10,
            "advice": "Do not eat if it smells bad or has mold."
        }
    }
    
    return info.get(stage, {
        "days_left": "Unknown",
        "freshness_score": 0,
        "advice": "Unable to estimate shelf life"
    })
    
def predict_banana(uploaded_file, model, class_names):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((160, 160))

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]) * 100)

    stage = class_names[predicted_index]
    life_info = get_life_info(stage)

    return {
        "stage": stage,
        "confidence": round(confidence, 2),
        "days_left": life_info["days_left"],
        "freshness_score": life_info["freshness_score"],
        "advice": life_info["advice"],
        "all_predictions": {
            class_names[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(class_names))
        }
    }