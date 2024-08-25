# Monkey Species Classification

This project is a Streamlit application that classifies images of monkeys into one of ten species using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The application allows users to upload an image, train the model, and see the training history and results.

## Project Overview

The objective of this project is to develop a machine learning application using Streamlit, which will include displaying media files, utilizing input widgets, showing progress and status updates, incorporating sidebars and containers, and visualizing data with graphs.

## Features

- **Image Classification**: Classifies images into one of ten monkey species.
- **Model Training**: Allows users to train the model with a specified number of epochs.
- **Progress and Status Updates**: Displays progress and status messages during model training.
- **Visualization**: Shows training and validation accuracy and loss graphs.
- **Input Widgets**: Includes sliders, buttons, and file upload input widgets.
- **Sidebars and Containers**: Utilizes Streamlit's sidebar for input widgets and containers for organizing different sections.

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- Git

### Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/WaruniR98/monkey-species-classification.git
    cd monkey-species-classification
    ```

2. **Create and activate a virtual environment**:

    On Windows:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

    On macOS/Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:

    ```bash
    streamlit run app3.py
    ```

## Usage

- **Training the Model**: Use the slider to select the number of epochs and click the "Train Model" button. The progress and status updates will be shown, along with training history graphs.
- **Classifying an Image**: Upload an image using the file uploader. The application will display the image and the predicted species. If the uploaded image is not of a monkey that belongs to the 10 species, a message will be shown.

## Monkey Species

The following monkey species are included in the classification model:

1. **Alouatta palliata (The Mantled Howler)**
2. **Erythrocebus patas (Patas Monkey)**
3. **Cacajao calvus (Bald Uakari)**
4. **Macaca fuscata (Japanese Macaque)**
5. **Cebuella pygmaea (Pygmy Marmoset)**
6. **Cebus capucinus (White-headed Capuchin)**
7. **Mico argentatus (Silvery Marmoset)**
8. **Saimiri sciureus (Common Squirrel Monkey)**
9. **Aotus nigriceps (Night Monkey)**
10. **Trachypithecus johnii (Nilgiri Langur)**

## Screenshots

![Training the Model](https://github.com/user-attachments/assets/cecf05b1-6117-4146-b8a2-33b24456f82f)
![Verification the Model](https://github.com/user-attachments/assets/b9a7a656-3901-4c5f-bd87-4cfc2c955f7c)
![Classifying an Image2](https://github.com/user-attachments/assets/4ca360f7-c4c2-4c61-939f-dbfbeda19836)
![Classifying an Image](https://github.com/user-attachments/assets/a8b0f6d4-3e72-4047-aaf2-ce4c72a585a0)

## Kaggle Dataset

Download it (https://www.kaggle.com/datasets/slothkong/10-monkey-species).

## Deployment

The application is deployed on Streamlit Cloud. Access it (https://monkey-species-classification-webapp-gzskggtkrxfnqgtnbeggtf.streamlit.app/).

## Repository

GitHub Repository: [Monkey Species Classification](https://github.com/Waruni9810/monkey-species-classification-webapp)
