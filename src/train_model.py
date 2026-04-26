import json

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input


def load_tr_ts_data():
    """
    Configures data generators for training and testing datasets.
    Applies data augmentation to the training set to improve model generalization.

    Returns:
        tuple: (train_generator, test_generator, labels_dictionary)
    """
    # Configure augmentation ONLY for training data to prevent overfitting
    # We use VGG16-specific preprocessing instead of simple 1./255 rescaling
    tr_data = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,                      # Random rotations up to 20 degrees
        width_shift_range=0.2,                  # Horizontal shift (20% of width)
        height_shift_range=0.2,                 # Vertical shift (20% of height)
        shear_range=0.2,                        # Image shift (tilting)
        zoom_range=0.2,                         # Random zoom in/out
        horizontal_flip=True,                   # Mirror reflection (left to right)
        fill_mode = 'nearest'                   # How to fill empty pixels after rotations
    )

    # Load training images from directory
    train_data = tr_data.flow_from_directory(
        directory="./dataset/training",
        target_size=(224,224),                  # VGG16 standard input size
        batch_size=32,                          # data volume (packet size - 16, 32,64...)
        class_mode='categorical'                # Multi-class classification (One-Hot)
    )

    # For validation/testing data: Only preprocessing, NO augmentation
    ts_data = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_data = ts_data.flow_from_directory(
        directory="./dataset/testing",
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical'
    )

    # Map directory names to class indices: {0: 'cat', 1: 'dog', ...}
    labels = {v: k for k, v in test_data.class_indices.items()}

    return train_data, test_data, labels


def create_model():
    """
    Initializes the model architecture using VGG16 as a frozen base.
    Adds a custom classification head on top of the convolutional base.

    Returns:
        keras.Sequential: Compiled Keras model.
    """
    # Load VGG16 pre-trained on ImageNet without the top classification layers
    VGG = VGG16(input_shape=(224,224,3), include_top=False, weights="imagenet")

    # Freeze VGG16 weights to keep pre-trained knowledge intact
    VGG.trainable = False

    # Define the new model structure
    model = keras.Sequential([
        VGG,                                                # Frozen convolutional base
        keras.layers.Flatten(),                             # Convert 2D feature maps to 1D vector
        keras.layers.Dense(units=256, activation="relu"),   # Fully connected layer
        keras.layers.Dense(units=256, activation="relu"),   # Extra depth for learning
        keras.layers.Dense(units=4, activation="softmax")   # Output layer (4 animal classes)
    ])

    # Compile with Adam optimizer and Categorical Crossentropy for multi-class tasks
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

    return model


def model_train(model, train_data, test_data, epochs):
    """
    Starts the model fitting process and saves the final result.

    Args:
        model (keras.Model): The model to be trained.
        train_data (DirectoryIterator): Generator for training images.
        test_data (DirectoryIterator): Generator for validation images.
        epochs (int): Number of training iterations.

    Returns:
        hist: Keras history object containing training metrics.
    """
    hist = model.fit(
        x=train_data,
        validation_data = test_data,
        #validation_split = 0.2,
        epochs = epochs
    )

    # Save the model in the latest Keras format
    model.save("model.keras")
    print("Training Complete!, Model saved as 'model.keras'")
    return hist


def main(epochs=10):
    """
    Main execution pipeline: load data, save labels, create and train the model.

    Args:
        epochs (int): Number of epochs to train, passed from UI or CLI.
    """
    # 1. Load data and extract class labels
    train_data, test_data, labels = load_tr_ts_data()

    # 2. Store labels in a JSON file for the Streamlit UI to use
    with open("config/labels.json", "w") as f:
        json.dump(labels, f)

    # 3. Build the model architecture
    model = create_model()

    # 4. Start the training process
    hist = model_train(model, train_data, test_data, epochs=epochs)

    # Saving history to a File
    with open("config/history.json", "w") as f:
        json.dump(hist.history, f)
    return hist


if __name__ == "__main__":
    # Standard entry point when running the script directly
    main(epochs=10)
