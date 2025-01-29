Overview
This project focuses on American Sign Language (ASL) recognition using a Random Forest Classifier. It utilizes Mediapipe for hand tracking and landmark extraction, with a machine learning model trained to classify different ASL signs.

Features
- Hand tracking using Mediapipe.
- Feature extraction from hand landmarks.
- Random Forest Classifier for sign classification.
- Dataset processing and model training pipeline.
- Prediction script for real-time ASL recognition.

Project Structure
├── create_dataset.py      # Extracts hand landmarks and creates dataset
├── process_pickle.py      # Processes dataset for consistency
├── train.py               # Trains the Random Forest model
├── app_1.py               # Runs prediction using the trained model
├── data.pickle            # Preprocessed dataset
├── model_1.p              # Trained Random Forest model
└── README.md              # Project documentation

Installation

Prerequisites
Ensure you have Python installed. Then, install the required dependencies:
bash:
pip install mediapipe opencv-python matplotlib scikit-learn numpy

Usage
1. Create Dataset
Run "create_dataset.py" to extract hand landmarks and generate a dataset.
bash:python create_dataset.py
This script will save a "data.pickle" file containing extracted features and labels.

2. Process Dataset (Optional)
To ensure the dataset is consistent, run:
bash:python process_pickle.py
This step processes and verifies the dataset structure.

 3. Train the Model
Train the Random Forest classifier using:
bash:python train.py
The trained model will be saved as "model_1.p".

4. Run Prediction
Use "app_1.py" to predict ASL signs in real-time:
bash:app_1.py
This script loads the trained model and uses Mediapipe for hand recognition.

 Model Details
- Classifier: Random Forest
- Input: Normalized hand landmark positions
- Output: Predicted ASL sign label

Dataset: 
American Sign Language (ASL) alphabet from kaggle was used for trainning
The Dataset has been broadly classified into Training and Testing Data. Training Data has been classified and segregated into 29 classes, of which 26 alphabets A-Z and 3 other classes of SPACE, DELETE, NOTHING. The test data set contains a mere 29 images, to encourage the use of real-world test images.
link:https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset

License
This project is open-source. Feel free to use and modify it as needed.

Author
Your Name : Prashuna B

