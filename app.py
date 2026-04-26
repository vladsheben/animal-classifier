"""
VGG16 Animal Classifier - Streamlit Application

This application provides an end-to-end interface for:
- Training a convolutional neural network (VGG16-based)
- Uploading and classifying animal images
- Visualizing prediction probabilities
- Monitoring training performance (accuracy & loss)
- Storing and retrieving training history from Supabase

Main Features:
--------------
1. Model Management:
   - Load trained model from disk
   - Retrain model via UI
   - Cache model for performance

2. Image Classification:
   - Upload JPG/PNG images
   - Preprocess images for VGG16
   - Display prediction with confidence

3. Analytics Dashboard:
   - Training metrics (accuracy/loss)
   - Visualization of learning curves
   - Overfitting detection hints

4. Database Integration:
   - Save training history to Supabase
   - Retrieve and display historical runs

Technologies:
-------------
- Streamlit (UI)
- TensorFlow / Keras (Deep Learning)
- NumPy / Pandas (Data Processing)
- Supabase (Cloud Database)
"""

import streamlit as st
import numpy as np
import os
import json
import pandas as pd
import logging.config

from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from pathlib import Path

from src import train_model, sp_db

os.chdir(Path(__file__).parent)


# Logging
logging.config.fileConfig("config/logging_config.ini")
logger = logging.getLogger()

# Page configuration
st.set_page_config(page_title="VGG16 Classifier", layout="wide")
st.title("🐾 Animal classifier (VGG16)")


# --- 1. Model Caching ---
@st.cache_resource
def get_model():
    """
    Load a trained Keras model from disk.
    Uses Streamlit resource caching to ensure the model is loaded only once
    and reused across reruns, improving performance.

    Returns: keras.Model or None: Loaded model if file exists, otherwise None.
    """
    MODEL_PATH = "model.keras"
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    return None


# --- 2. Label-Caching ---
@st.cache_data
def load_labels():
    """
    Loads class labels from the JSON file created during training.
    Using @st.cache_data for lightweight data structures like dictionaries.

    Returns: dict or None: Dictionary mapping class index to label name.
    """
    if os.path.exists("config/labels.json"):
        with open("config/labels.json", "r") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return None


# --- 3. Loads history from the JSON file created during training. ---
@st.cache_data
def load_training_history():
    """
    Load model training history from JSON file.
    The history contains training and validation metrics
    such as accuracy and loss over epochs.

    Returns: dict or None: Dictionary containing training metrics.
    """
    if os.path.exists("config/history.json"):
        with open("config/history.json", "r") as f:
            return json.load(f)
    return None


# --- 4. Streamlit (UI LOGIC & PREDICTIONS) ---

# Initialize session state for analysis and training lock if not present
st.session_state.setdefault('is_analyzing', False)
st.session_state.setdefault('is_training', False)

# Initialize session state from the cached Json file
if 'training_history' not in st.session_state:
    st.session_state.training_history = load_training_history()

# Load model and labels at the start of the app
model = get_model()
LABELS = load_labels()

with st.sidebar:
    st.header(":gear: Settings")

    # User-adjustable hyperparameter for training
    epochs = st.slider("Number of epochs:", min_value=2, max_value=100, value=5)

    # If the training is NOT running, show the button
    if not st.session_state.is_training:
        if st.button(":rocket: Train Model", disabled=st.session_state.is_analyzing):
            st.session_state.is_training = True
            st.rerun()

    # If the status True, we start the training process
    else:
        with st.spinner(f'Training for {epochs} epochs in progress...'):
            # 1. launch training and get history
            hist = train_model.main(epochs)
            # 2. Saving history to a session
            st.session_state.training_history = hist.history
            # 3 Saving Training history FinalProject to a supabase table
            sp_db.main(hist.history)

        st.success('Training Complete!')

        # Reset state and clear cache to reload the new model/labels
        st.session_state.is_training = False
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    # Sidebar Status Information
    if model and LABELS:
        st.divider() # Horizontale Linie
        st.success(":heavy_check_mark: Model is Loaded")

        # Classes
        with st.expander(f"Classes: {len(LABELS)}", expanded=False):
            for i, name in LABELS.items():
                st.write(f"— {name.capitalize()}")
    else:
        st.error("⚠️ Model not found!!!")

# Main Interface Logic
if model is None or LABELS is None:
    st.warning("⚠️ Model not found. Please click 'Train Model' in the sidebar to start.")
else:
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a photo (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file:   # is not None:
        logging.info(f"File uploaded: {uploaded_file.name}")

        # We create columns, Image on the left, Results on the right
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Your image", use_container_width=True)

        # --- Prediction Block ---
        with st.spinner("Analyzing..."):
            # --- Image Preprocessing for VGG16 ---
            img = load_img(uploaded_file, target_size=(224, 224))
            img_array = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

            # 1. Get model probabilities
            predictions = model.predict(img_array)[0]
            # 2. Get index of the class with highest probability
            winner_idx = np.argmax(predictions)
            # 3. Retrieve class name from labels dictionary
            label = LABELS.get(winner_idx, "Unknown")
            # 4. Calculate confidence percentage
            confidence = predictions[winner_idx] * 100

        with col2:
            # Display main result as a metric
            st.subheader(":calendar: Prediction result:")
            st.metric("Class", label.capitalize(), f"{confidence:.2f}% confidence")

            # --- Probability Visualization ---
            with st.expander(":mag: Probabilities by class, show more details", expanded=False):
                for i, prob in enumerate(predictions):
                    #class_name = LABELS.get(i, f"Class {i}")
                    st.write(f"{LABELS.get(i, f'Class {i}').capitalize()}:")
                    # Show confidence bar for each class
                    st.progress(float(prob))


    # --- Analytics Block ---
    with (st.expander(":chart_with_upwards_trend: See Training History & Analytics", expanded=False)):
        # 2. TRAINING METRICS (Displayed below the uploader, independent of image upload)
        h = st.session_state.training_history
        if h:
            #st.divider()
            st.subheader(":bar_chart: Last Training Results")

            # --- NEW: Summary Statistics Metrics ---
            mc_1, mc_2, mc_3, mc_4 = st.columns(4)

            # Extract best values from history
            best_acc = max(h['accuracy'])
            best_val_acc = max(h['val_accuracy'])
            min_loss = min(h['loss'])
            min_val_loss = min(h['val_loss'])

            # Calculate the difference (gap) between training and validation accuracy
            gap = best_val_acc - best_acc
            overfit_gap = best_acc - best_val_acc
            delta_text = f"{gap:+.2%} gap"

            # Color logic accepts:'off', or a color name ('orange', 'yellow', 'violet', 'gray'/'grey', 'primary')
            if overfit_gap > 0.10:
                d_color = "normal"
            elif overfit_gap > 0:
                d_color = "inverse"
            elif overfit_gap == 0:
                d_color = "off"     # "blue"
            else:
                d_color = "normal"

            with mc_1:
                st.metric("Max Train Accuracy", f"{best_acc:.2%}")
            with mc_2:
                st.metric("Max Val Accuracy", f"{best_val_acc:.2%}", delta_text, delta_color=d_color,
                          help="The optimal gap between Train and Val is less than 10%")
            with mc_3:
                st.metric("Min Train Loss", f"{min_loss:.4f}")
            with mc_4:
                st.metric("Min Val Loss", f"{min_val_loss:.4f}")

            if overfit_gap > 0.10:
                st.warning("⚠️ **Warning: Overfitting!** Accuracy during training is significantly higher."
                           "It is recommended to add augmentation or reduce the number of epochs.")

            st.divider()

            graph_col1, graph_col2 = st.columns(2)

            with graph_col1:
                # Accuracy Chart
                st.write("**Accuracy History**")
                acc_df = pd.DataFrame({
                    "Train": h['accuracy'],
                    "Val": h['val_accuracy']
                })
                st.line_chart(acc_df)
                st.caption("light Blue: Training, Blue: Validation")

            with graph_col2:
                # Loss Chart
                st.write("**Loss History**")
                loss_df = pd.DataFrame({
                    "Train": h['loss'],
                    "Val": h['val_loss']
                })
                st.line_chart(loss_df)
                st.caption("light Blue: Training, Blue: Validation")

            st.info("💡 **Tip:** If Max Train Accuracy is significantly higher than Max Val Accuracy (e.g., 98% vs. 70%), "
                    "this is a visual signal of overfitting. In this case, it's worth reducing the number of epochs"
                    " or adding more augmentation to Train Model.py.")

            # --- Global history table from Supabase ---
            st.divider()
            st.subheader(":trophy: Training History (Supabase)")

            if st.button("🔄 Refresh History"):
                try:
                    client = sp_db.get_client()
                    db_data = sp_db.fetch_history(client)

                    if db_data:
                        df = pd.DataFrame(db_data)

                        # Convert to date format and add 2 hours
                        df['created_at'] = pd.to_datetime(df['created_at']) + pd.to_timedelta(2, unit='h')

                        # Rename columns for display
                        df = df[['created_at', 'epochs_count', 'accuracy', 'val_accuracy', 'loss', 'val_loss']]
                        df = df.rename(columns={
                            "created_at": "Date of creation",
                            "epochs_count": "Epochs",
                            "accuracy": "Max Accuracy (Tr)",
                            "val_accuracy": "Max Accuracy (Val)",
                            "loss": "Min Loss (Tr)",
                            "val_loss": "Min Loss (Val)"
                        })

                        st.dataframe(
                            df.sort_values('Date of creation', ascending=False).style.format({
                                "Max Accuracy (Tr)": "{:.2%}",
                                "Max Accuracy (Val)": "{:.2%}",
                                "Min Loss (Tr)": "{:.4f}",
                                "Min Loss (Val)": "{:.4f}",
                                "Date of creation": lambda x: x.strftime('%Y-%m-%d %H:%M')
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No history found in database.")
                except Exception as e:
                    st.error(f"Could not load history: {e}")
                    st.info("Please check your Supabase connection and try again.")

        else:
            st.info("No training history available.")
