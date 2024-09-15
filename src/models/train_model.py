import joblib
from model import get_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data (X = [height], y = weight)
X = np.array(
    [
        [140],
        [160],
        [169],
        [180],
        [169],
        [160],
        [155],
        [158],
        [145],
        [150],
        [165],
        [165],
        [162],
        [160],
        [150],
        [165],
        [156],
        [152],
        [150],
        [150],
        [153],
        [180],
    ]
)

y = np.array(
    [
        40,
        55,
        60,
        65,
        60,
        42,
        56,
        45,
        34,
        43,
        60,
        52,
        46,
        45,
        40,
        55,
        39,
        42,
        41,
        41,
        40,
        75,
    ]
)  # Weights in kg
# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Get the model from model.py
model = get_model()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(np.sqrt(mse))

# Save the trained model
joblib.dump(model, "saved_models/linear_regression_model.pkl")
