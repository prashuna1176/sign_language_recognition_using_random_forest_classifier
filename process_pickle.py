import pickle

# Load your original data
file_path = r"D:\isl\new\data.pickle"

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Assuming the data is a tuple or list with two main parts
images, labels = data  # Unpack the dataset

# Create a homogeneous dictionary
homogeneous_data = {
    "images": images,  # Array of image data
    "labels": labels   # Corresponding labels
}

# Save the dictionary to a new pickle file
homogeneous_file_path = 'homogeneous_data.pickle'

with open(homogeneous_file_path, 'wb') as file:
    pickle.dump(homogeneous_data, file)

print(f"Homogeneous pickle file saved as: {homogeneous_file_path}")

# Verify the saved file
with open(homogeneous_file_path, 'rb') as file:
    verified_data = pickle.load(file)

# Print verification result
print("Verified data structure:")
print("Keys:", verified_data.keys())
print("Number of images:", len(verified_data["images"]))
print("Number of labels:", len(verified_data["labels"]))
