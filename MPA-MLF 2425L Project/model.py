# -----------------------------
# Import necessary libraries
# -----------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import keras_tuner as kt

# -----------------------------
# Define constants and file paths
# -----------------------------
MODEL_PATH   = "best_model.h5"
TRAIN_OUTPUT = "trainoutput.csv"
TEST_OUTPUT  = "testoutput.csv"
LABEL_FILE   = "label_train.csv"

# -----------------------------
# 1. Load and preprocess training data
# -----------------------------
def load_train_data(train_dir="Train/", label_file=LABEL_FILE):
    label_df = pd.read_csv(label_file)  # Read labels CSV
    X_list, y_list, id_list = [], [], []
    
    # Loop through each label row
    for _, row in label_df.iterrows():
        fn = str(row["ID"])
        if not fn.endswith(".npy"):
            fn += ".npy"  # Ensure file has .npy extension
        arr = np.load(os.path.join(train_dir, fn))  # Load numpy array
        X_list.append(arr)
        y_list.append(int(row["target"]))  # Store label
        id_list.append(fn)  # Store file name for output reference

    # Normalize and reshape data for model input
    X = np.expand_dims(np.array(X_list), -1).astype("float32") / 255.0
    y = np.array(y_list)
    return X, y, id_list

# -----------------------------
# 2. Train-validation split and compute class weights
# -----------------------------
def prepare_data(X, y, test_size=0.2, random_state=42):
    # Split with stratified sampling to preserve class ratios
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Compute balanced class weights to handle imbalanced classes
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_tr),
        y=y_tr
    )
    cw = {i: weights[i] for i in range(len(weights))}
    print("Class Weights:", cw)
    return X_tr, X_val, y_tr, y_val, cw

# -----------------------------
# 3. Define model structure for hyperparameter tuning
# -----------------------------
def model_builder(hp):
    model = keras.Sequential([
        layers.Input((72, 48, 1)),
        # First convolutional layer
        layers.Conv2D(hp.Int("filters_1", 8, 64, step=8),
                      (hp.Choice("kernel_size", [3,5]),
                       hp.Choice("kernel_size", [3,5])),
                      activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Second convolutional layer
        layers.Conv2D(hp.Int("filters_2", 8, 64, step=8),
                      (hp.Choice("kernel_size", [3,5]),
                       hp.Choice("kernel_size", [3,5])),
                      activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(hp.Float("dropout", 0.0, 0.5, step=0.1)),
        layers.Dense(3, activation="softmax"),  # 3-class classification
    ])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("lr", [1e-3,3e-4,1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# 4. Tune hyperparameters and train the best model
# -----------------------------
def train_model(X_tr, y_tr, X_val, y_val, cw):
    # Use random search tuner
    tuner = kt.RandomSearch(
        model_builder,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        directory="my_tuner_dir",
        project_name="5G_base_stations"
    )

    # Start hyperparameter search
    tuner.search(
        X_tr, y_tr,
        epochs=20,
        validation_data=(X_val, y_val),
        class_weight=cw,
        verbose=1
    )

    # Get best hyperparameters and train final model
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("Best HPs:", {k: best_hps.get(k) for k in best_hps.values.keys()})

    model = tuner.hypermodel.build(best_hps)
    model.fit(
        X_tr, y_tr,
        epochs=30,
        validation_data=(X_val, y_val),
        class_weight=cw,
        verbose=1
    )
    return model

# -----------------------------
# 5. Predict and save train/test results
# -----------------------------
def generate_outputs(model, X, id_list, test_dir="Test/"):
    # Generate predictions for training data
    preds  = model.predict(X)
    labels = np.argmax(preds, axis=1)
    ids    = [os.path.splitext(fn)[0] for fn in id_list]

    # Save train predictions
    pd.DataFrame({"ID": ids, "target": labels}) \
      .to_csv(TRAIN_OUTPUT, index=False)
    print(f"Saved train predictions to {TRAIN_OUTPUT}")

    # Load and preprocess test data
    test_files = sorted(os.listdir(test_dir))
    X_test = []
    for fn in test_files:
        arr = np.load(os.path.join(test_dir, fn))
        X_test.append(np.expand_dims(arr, -1).astype("float32")/255.0)
    X_test = np.array(X_test)

    # Predict and save test results
    preds_t    = model.predict(X_test)
    labels_t   = np.argmax(preds_t, axis=1)
    ids_test   = [os.path.splitext(fn)[0] for fn in test_files]
    pd.DataFrame({"ID": ids_test, "target": labels_t}) \
      .to_csv(TEST_OUTPUT, index=False)
    print(f"Saved test predictions to {TEST_OUTPUT}")

# -----------------------------
# 6. Compare predictions to true labels
# -----------------------------
def get_match_stats():
    pred = pd.read_csv(TRAIN_OUTPUT)
    true = pd.read_csv(LABEL_FILE)[["ID","target"]]
    merged = pred.merge(true, on="ID", suffixes=("_pred","_true"))

    # Calculate matching predictions
    total   = len(merged)
    matches = (merged["target_pred"] == merged["target_true"]).sum()
    return matches, total, matches / total

# -----------------------------
# 7. Main training and evaluation loop
# -----------------------------
def main():
    # Load and prepare the data
    X, y, id_list       = load_train_data()
    X_tr, X_val, y_tr, y_val, cw = prepare_data(X, y)

    # Load existing model or train a new one
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
    else:
        print("No saved model—training initial model")
        model = train_model(X_tr, y_tr, X_val, y_val, cw)
        model.save(MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")

    best_score = 0.0
    attempt    = 1

    # Loop until we get a perfect match on the training set
    while True:
        print(f"\n=== Attempt #{attempt}: Generating outputs ===")
        generate_outputs(model, X, id_list)

        matches, total, ratio = get_match_stats()
        print(f"Match ratio: {matches}/{total} = {ratio*100:.2f}%")

        # Save model if it's the best so far
        if ratio > best_score:
            best_score = ratio
            model.save(MODEL_PATH)
            print(f"New best model saved (ratio {ratio*100:.2f}%)")

        # Stop training if perfect match achieved
        if ratio >= 1.0:
            print(f"Perfect match achieved on attempt #{attempt}.")
            break

        # Otherwise, retrain and try again
        print(f"Not perfect—retraining for attempt #{attempt+1}…")
        model = train_model(X_tr, y_tr, X_val, y_val, cw)
        attempt += 1

    print("\nDone. Best model is saved at", MODEL_PATH)

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
