import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model

# Set dataset path
train_dir = 'monkey_dataset/training'
valid_dir = 'monkey_dataset/validation'
model_path = 'monkey_species_model.keras'

# Create the model using transfer learning
@st.cache_resource
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)  # 10 classes for monkeys

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load existing model or create a new one
def load_or_create_model():
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model()
    return model

model = load_or_create_model()

def load_data(train_dir, valid_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, valid_generator

train_generator, valid_generator = load_data(train_dir, valid_dir)

# CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #E7E8D8;
        color: #155724;
        font-family: 'Arial', sans-serif;
    }
    
.stButton>button {
    background-color: #436850;
    color: white;
    border-radius: 5px;
}
.stSidebar .stButton>button {
    background-color: #28a745;
    color: white;
    border-radius: 5px;
}
.stTextInput>div>input {
    background-color: #e9f7ef;
    color: #155724;
    border-radius: 5px;
}
.stMarkdown h1 {
    color: #155724;
    text-align: center;
}
.stMarkdown h2 {
    color: #155724;
    text-align: center;
}
.header {
    padding: 20px 0;
    text-align: center;
}
.header h1 {
    margin: 0;
    font-size: 3em;
}
.header h2 {
    margin: 0;
    font-size: 1.5em;
}
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div class="header">
        <h1>Monkey Species Classification</h1>
        <h2>Using CNN and TensorFlow</h2>
    </div>
    """,
    unsafe_allow_html=True
)

def main():
    st.sidebar.header('Training Parameters')
    epochs = st.sidebar.slider('Number of epochs', 1, 50, 25)

    if st.sidebar.button('Train Model'):
        with st.spinner('Training the model...'):
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=valid_generator,
                callbacks=[early_stopping, reduce_lr]
            )
            model.save(model_path)  # Save the trained model in .keras format
            st.success('Model trained and saved successfully!')

            st.write("### Training History")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['accuracy'], label='Training Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()

            ax2.plot(history.history['loss'], label='Training Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()

            st.pyplot(fig)

    st.sidebar.header('Predict Image')
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = plt.imread(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        img = load_img(uploaded_file, target_size=(150, 150))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        pred = model.predict(img)
        pred_class = np.argmax(pred, axis=1)
        confidence = np.max(pred)

        species = list(train_generator.class_indices.keys())[pred_class[0]]

        species_info = {
            'n0': ("Alouatta palliata (The Mantled Howler)", "The mantled howler is a species of howler monkey, a type of New World monkey, from Central and South America. They are known for their loud howls, which can travel for miles through dense forest."),
            'n1': ("Erythrocebus patas (Patas Monkey)", "The patas monkey, also known as the wadi monkey or hussar monkey, is a ground-dwelling monkey distributed over West Africa, and into East Africa. It is the fastest runner among primates."),
            'n2': ("Cacajao calvus (Bald Uakari)", "The bald uakari is a small South American primate characterized by its bright red face, bald head, and long coat of hair on its body. They live in the Amazon rainforest and have a diet mainly consisting of seeds."),
            'n3': ("Macaca fuscata (Japanese Macaque)", "The Japanese macaque, also known as the snow monkey, is a terrestrial Old World monkey species native to Japan. They are known for their habit of bathing in hot springs in the winter."),
            'n4': ("Cebuella pygmaea (Pygmy Marmoset)", "The pygmy marmoset is a small species of monkey native to rainforests of the western Amazon Basin in South America. They are notable for being the smallest monkeys in the world."),
            'n5': ("Cebus capucinus (White-headed Capuchin)", "The white-headed capuchin, also known as the white-faced capuchin or white-throated capuchin, is a medium-sized New World monkey of the family Cebidae, native to Central America and the extreme north-western portion of South America."),
            'n6': ("Mico argentatus (Silvery Marmoset)", "The silvery marmoset is a New World monkey that lives in the eastern Amazon Rainforest in Brazil. These small monkeys have a silvery fur and are highly social animals."),
            'n7': ("Saimiri sciureus (Common Squirrel Monkey)", "The common squirrel monkey is a small primate found in the tropical forests of Central and South America. They are highly social and live in large groups."),
            'n8': ("Aotus nigriceps (Night Monkey)", "The black-headed night monkey, also known as the owl monkey, is a nocturnal New World monkey native to Panama and Colombia. They are known for their large eyes adapted for night vision."),
            'n9': ("Trachypithecus johnii (Nilgiri Langur)", "The Nilgiri langur is a primate found in the Western Ghats of South India. They are distinguished by their glossy black fur and brownish crown.")
        }

        confidence_threshold = 0.5  # Set a threshold for confidence
        if confidence < confidence_threshold:
            st.write("The uploaded image is not of a monkey that belongs to the 10 species. Try again.")
        elif species in species_info:
            st.write(f"**{species_info[species][0]}**")
            st.write(f"{species_info[species][1]}")
        else:
            st.write("The uploaded image is not of a monkey that belongs to the 10 species. Try again.")

if __name__ == "__main__":
    main()
