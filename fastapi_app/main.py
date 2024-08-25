import tensorflow as tf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = FastAPI()

class ImagePath(BaseModel):
    image_path: str

# Load the trained model
model = tf.keras.models.load_model('../monkey_species_model.keras')

# List of monkey species corresponding to the model's output
monkey_species = [
    "Alouatta palliata (The Mantled Howler)",
    "Erythrocebus patas (Patas Monkey)",
    "Cacajao calvus (Bald Uakari)",
    "Macaca fuscata (Japanese Macaque)",
    "Cebuella pygmaea (Pygmy Marmoset)",
    "Cebus capucinus (White-headed Capuchin)",
    "Mico argentatus (Silvery Marmoset)",
    "Saimiri sciureus (Common Squirrel Monkey)",
    "Aotus nigriceps (Night Monkey)",
    "Trachypithecus johnii (Nilgiri Langur)"
]

# Set a confidence threshold to determine if the image does not belong to any species
CONFIDENCE_THRESHOLD = 0.6

def predict_species(image_path: str):
    try:
        img = load_img(image_path, target_size=(150, 150))  # Resize image
        img_array = img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions

        # Normalize pixel values (important if your model expects values in range [0, 1])
        img_array = img_array / 255.0

        # Get raw predictions
        predictions = model.predict(img_array)

        # Apply softmax to get probabilities (if not included in your model)
        predictions = tf.nn.softmax(predictions[0]).numpy()

        # Get the index of the highest probability
        predicted_index = np.argmax(predictions)
        predicted_confidence = predictions[predicted_index]
        
        # Check if the confidence is above the threshold
        if predicted_confidence < CONFIDENCE_THRESHOLD:
            return "The uploaded image is not of a monkey that belongs to the 10 species. Try again."
        
        predicted_species = monkey_species[predicted_index]
        return predicted_species
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(image: ImagePath):
    predicted_species = predict_species(image.image_path)
    return {"predicted_species": predicted_species}
