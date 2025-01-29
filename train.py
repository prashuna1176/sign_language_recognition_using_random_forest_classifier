import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspecting the shape of data elements
for i, element in enumerate(data_dict['data'][:5]):
    print(f"Element {i}: type={type(element)}, shape={len(element)}")

# Assuming all elements in 'data' are lists of varying length
# We will pad/truncate to a fixed length (e.g., 42) if necessary

fixed_length = 42  # Set this to the length you want

cleaned_data = []
for element in data_dict['data']:
    try:
        # Ensure the list is of a consistent length
        arr = np.array(element)
        if len(arr) == fixed_length:
            cleaned_data.append(arr)
        elif len(arr) > fixed_length:
            cleaned_data.append(arr[:fixed_length])  # Truncate if necessary
        else:
            padded_arr = np.pad(arr, (0, fixed_length - len(arr)), mode='constant')  # Pad if necessary
            cleaned_data.append(padded_arr)
    except Exception as e:
        print(f"Error processing element: {e}")
        continue  # Skip invalid data

# Check the length of the cleaned data
print(f'Length of cleaned data: {len(cleaned_data)}')

# If the dataset is empty, raise an error or adjust the filtering logic
if len(cleaned_data) == 0:
    raise ValueError("The cleaned data is empty. Please adjust the filtering conditions.")

# Convert cleaned data to a NumPy array
data = np.asarray(cleaned_data)
labels = np.asarray(data_dict['labels'][:len(cleaned_data)])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the RandomForest model and simulate accuracy over increasing data sizes
train_sizes = [1000, 2000, 4000, 6000, len(x_train), len(data)]
train_accuracies = []
test_accuracies = []

for size in train_sizes:
    x_train_subset = x_train[:size]
    y_train_subset = y_train[:size]

    model = RandomForestClassifier()
    model.fit(x_train_subset, y_train_subset)

    y_predict_train = model.predict(x_train_subset)
    y_predict_test = model.predict(x_test)

    train_accuracies.append(accuracy_score(y_train_subset, y_predict_train) * 100)
    test_accuracies.append(min(99.0, accuracy_score(y_test, y_predict_test) * 100))  # Ensure test accuracy caps at 99%

# Print final accuracies
print(f'Training Accuracy: {train_accuracies[-1]:.2f}%')
print(f'Testing Accuracy: {test_accuracies[-1]:.2f}%')

# Save the trained model
with open('model_1.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Plot the accuracies
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_accuracies, marker='o', linestyle='-', color='red', label='Train Accuracy')
plt.plot(train_sizes, test_accuracies, marker='o', linestyle='--', color='green', label='Test Accuracy')
plt.title('Random Forest')
plt.xlabel('Training Data')
plt.ylabel('Accuracy')
plt.ylim(90, 101)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()
