from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()

# Load your trained model
model = load_model('../monkey_species_model.keras')

class PredictionResult(BaseModel):
    message: str
    species: str = None
    confidence: float = None

@app.post("/predict/", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    # Load and preprocess the image
    image = Image.open(BytesIO(await file.read()))
    image = image.resize((150, 150))  # Resize to match model input
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Preprocess the image

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Threshold for considering a valid prediction
    confidence_threshold = 0.5

    # Map class indices to actual monkey names
    species_info = [
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

    if confidence < confidence_threshold:
        return PredictionResult(message="The uploaded image is not of a monkey that belongs to the 10 species. Try again.")
    
    species = species_info[predicted_class] if predicted_class < len(species_info) else "Unknown"
    
    return PredictionResult(message="Prediction successful.", species=species, confidence=float(confidence))

