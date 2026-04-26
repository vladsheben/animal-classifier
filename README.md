## Project
# 🐾 Animal Classifier (VGG16) – Streamlit App

A full-stack machine learning web application for training, evaluating, and using an image classification model based on VGG16.

Built with **Streamlit**, **TensorFlow/Keras**, and **Supabase** for storing training history.

---

## 📌 Features

### 🔍 Image Classification
- Upload images (JPG, PNG)
- Automatic preprocessing for VGG16
- Real-time predictions with confidence score
- Probability breakdown per class

### 🧠 Model Training
- Train model directly from UI
- Adjustable number of epochs
- Training progress feedback
- Automatic model reload after training

### 📊 Analytics Dashboard
- Training vs validation accuracy
- Training vs validation loss
- Key metrics (max accuracy, min loss)
- Overfitting detection hints

### 🗄 Database Integration (Supabase)
- Store training results
- Retrieve training history
- Display historical runs in table format

---
## 🏗 Project Structure
Animal_Classifier/
│
├── app.py # Main Streamlit app
├── model.keras # Trained model (generated)
├── config/
│ ├── labels.json # Class labels
│ ├── history.json # Training history
│ └── logging_config.ini # Logging configuration
│
├── dataset/
│ ├── testing/ # Testing dataset
│ └── training/ # Training dataset
│
├── logs/
│ └── process.log # Logging
│
├── src/
│ ├── train_model.py # Model training logic
│ └── sp_db.py # Supabase integration
│
├── requirements.txt
└── README.md
---

## Installation
### 1. Clone repository
```bash
git clone https://github.com/vladsheben/animal-classifier.git
cd animal-classifier
````
### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run Application
```bash
streamlit run app.py
```
---

## 🧠 How It Works
1. Model Loading
- Loads trained model from model.keras
- Uses Streamlit caching for performance
2. Image Processing Pipeline
- Resize image to 224x224
- Convert to NumPy array
- Apply VGG16 preprocessing
- Feed into neural network
3. Prediction
- Outputs probability vector
- Selects class with highest probability
- Displays label + confidence
## 📊 Training Workflow
1. Click "Train Model" in sidebar
2. Model is trained using selected epochs
3. Training history is saved:
- locally (history.json)
- remotely (Supabase)
4. App reloads model automatically
## 📈 Example Output
- Predicted class: Dog
- Confidence: 92.34%
- Probability distribution for all classes
## 🛠 Technologies Used
- Python
- Streamlit
- TensorFlow / Keras (VGG16)
- NumPy / Pandas
- Supabase (PostgreSQL)
Logging (logging.config)
## ⚠️ Known Limitations
- Model performance depends on training data quality
- No GPU acceleration by default
- Limited data augmentation (can be improved)
## 🚀 Future Improvements
- Add model selection (ResNet, EfficientNet)
- Improve data augmentation
- Add batch image processing
- Deploy to cloud (Streamlit Cloud / AWS)
- Add authentication system
