# Pet Mood Detector

This project is a web application that uses a Convolutional Neural Network (CNN) to detect the mood of a pet (cat or dog) from an uploaded image.

## Features

*   Upload an image of a cat or dog.
*   The model will predict the pet's mood as one of: angry, happy, or puzzled.
*   The prediction is displayed to the user.

## Project Structure
```
pet-mood-detector/
├── .gitignore
├── models/
│   └── pet_mood_cnn.pth
├── requirements.txt
├── src/
│   ├── app.py
│   ├── mood_model.py
│   └── static/
├── templates/
│   └── index.html
└── dataset/
    ├── cat/
    │   ├── angry/
    │   ├── happy/
    │   └── puzzled/
    └── dog/
        ├── angry/
        ├── happy/
        └── puzzled/
```

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Cohegen/pet-mood-detector.git
    cd pet-mood-detector
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python src/app.py
    ```

5.  Open your web browser and navigate to `http://127.0.0.1:5000`.

## Model

The CNN model is trained on a dataset of cat and dog images, categorized by mood. The model is saved in the `models/` directory.

## Dataset

The dataset used to train the model is located in the `dataset/` directory. It is organized by pet type (cat/dog) and mood (angry/happy/puzzled).
