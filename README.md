import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Subjects and Semesters
subjects = ["Physics", "Maths", "History", "Urdu"]
semesters = ["Semester1", "Semester2", "Semester3", "Semester4"]

# Marks data
marks = {
    "Semester1": [34, 97, 89, 90],
    "Semester2": [64, 67, 39, 70],
    "Semester3": [46, 77, 79, 80],
    "Semester4": [89, 87, 59, 50],
}

# Convert to DataFrame
marks_array = np.array(list(marks.values()))
df = pd.DataFrame(marks_array, index=semesters, columns=subjects)

# Plot original data
df.plot(kind="bar", figsize=(10, 5))
plt.title("Semester-wise Subject Marks")
plt.xlabel("Semesters")
plt.ylabel("Marks")
plt.ylim(30, 100)
plt.show()

# Predict next semester marks using LSTM for each subject
predicted_marks = {}

for subject in subjects:
    # Normalize using pandas
    df[f"{subject}_scaled"] = (df[subject] - df[subject].min()) / (df[subject].max() - df[subject].min())
    
    # Prepare input for LSTM
    X = df[f"{subject}_scaled"].iloc[:-1].values.reshape(-1, 1, 1)
    y = df[f"{subject}_scaled"].iloc[1:].values.reshape(-1, 1)

    # Build and train LSTM model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(1, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)

    # Predict the next value
    last_scaled = df[f"{subject}_scaled"].iloc[-1]
    input_scaled = np.array([[last_scaled]]).reshape(1, 1, 1)
    pred_scaled = model.predict(input_scaled)

    # Convert back to actual marks
    subject_min = df[subject].min()
    subject_max = df[subject].max()
    predicted = subject_min + pred_scaled * (subject_max - subject_min)
    predicted_marks[subject] = predicted[0][0]

for subject in subjects:
    df.at["Semester5", subject] = predicted_marks[subject]


# Plot with prediction
df[subjects].plot(kind="bar", figsize=(10, 6))
plt.title("All Subjects Marks with 5th Semester Prediction")
plt.xlabel("Semesters")
plt.ylabel("Marks")
plt.ylim(0, 100)
plt.grid(True)
plt.xticks(rotation=0)
plt.show()

# Print predicted marks
print("\n📊 Predicted Marks for Semester 5:")
for subj in subjects:
    print(f"{subj}: {predicted_marks[subj]:.2f}")
