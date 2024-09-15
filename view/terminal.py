import joblib
import sys
import numpy as np

# Predefined model path
MODEL_PATH = "saved_models/linear_regression_model.pkl"


def load_model():
    """Loads the trained model from the predefined file path."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        sys.exit(1)


def make_prediction(model, input_value):
    """Makes a prediction using the loaded model and input data."""
    input_data = np.array([[input_value]])  # Reshape input as required by the model
    prediction = model.predict(input_data)
    return prediction[0]


if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python predict_from_terminal.py <input_value>")
        sys.exit(1)

    # Extract the input value from the command line
    input_value = float(sys.argv[1])  # Convert the input value (height) to a float

    # Load the model
    model = load_model()

    # Make a prediction
    predicted_weight = make_prediction(model, input_value)

    # Print the result
    print(f"Predicted weight for height {input_value} cm: {predicted_weight:.2f} kg")
