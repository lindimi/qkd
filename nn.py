import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("mdi_qkd_dataset.csv")

# Example columns (depends on how you want to structure inputs/outputs):
# Input features: Lbc_km, Y0, ed, N
# Target outputs: mu, nu, omega, p_mu, p_nu, p_omega

X = df[["Lbc_km", "Y0", "ed", "N"]].values
y = df[["mu", "nu", "omega", "p_mu", "p_nu", "p_omega"]].values
# Optionally, you could predict the key_rate as well; just add it to y if desired.

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(y.shape[1])  # final layer has as many neurons as output dimension
])

# Compile with mean-squared error (MSE) for regression
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']  # track mean absolute error
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=100,           # number of training passes
    batch_size=32,        # how many samples before updating weights
    validation_split=0.1, # slice out 10% of train set for validation
    verbose=1
)

# Example new sample
sample_input = np.array([[20, 1e-6, 0.01, 1e12]])  # shape (1,4)

prediction = model.predict(sample_input)  # shape (1,6)
mu_pred, nu_pred, omega_pred, pmu_pred, pnu_pred, pomega_pred = prediction[0]

print("Predicted mu:", mu_pred)
print("Predicted nu:", nu_pred)
print("Predicted omega:", omega_pred)
print("Predicted p_mu:", pmu_pred)
print("Predicted p_nu:", pnu_pred)
print("Predicted p_omega:", pomega_pred)