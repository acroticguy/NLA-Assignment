import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Loads training and test data from text files."""
    print("Loading data...")
    # Training data
    train_labels = np.loadtxt('dzip.txt', delimiter=',', dtype=np.int32)
    train_images = np.loadtxt('azip.txt', delimiter=',').T

    test_labels = np.loadtxt('dtest.txt', delimiter=',', dtype=np.int32)
    test_images = np.loadtxt('testzip.txt', delimiter=',').T

    # Normalize to [0, 1]
    train_images = (train_images + 1) / 2
    test_images = (test_images + 1) / 2

    print(f"Training data shape: {train_images.shape}")
    print(f"Test data shape: {test_images.shape}")
    return train_images, train_labels, test_images, test_labels

def map_data(images, labels):
    """Maps images to their corresponding labels."""
    image_map = {key: [] for key in range(10)}
    for i in range(len(labels)):
        label = labels[i]
        image_map[label].append(images[i])

    image_map = {key: np.array(value) for key, value in image_map.items()}
    return image_map

train_images, train_labels, test_images, test_labels = load_data()

train_map = map_data(train_images, train_labels)
test_map = map_data(test_images, test_labels)

svd_map = {}

for label, images in train_map.items():
    U, S, Vt = np.linalg.svd(images.T, full_matrices=False)
    
    # Store the first 20 singular vectors for each class
    svd_map[label] = U[:, :20]
    print(f"Class {label} SVD shape: {svd_map[label].shape}")

k_accuracy_scores = {i: {} for i in range(20, 5, -1)}

for i in range(20, 5, -1):
    accuracy_scores = [0 for _ in range(10)]
    test_pred = {key: [] for key in range(10)}
    for key, images in test_map.items():
        correct_count = 0
        for image in images:
            residual_norm = []
            for j in range(10):
                projection = svd_map[j][:, :i] @ svd_map[j][:, :i].T @ image
                residual_norm.append(np.linalg.norm(image - projection))
            residual_norm = np.array(residual_norm)
            predicted_label = np.argmin(residual_norm)
            test_pred[key].append(predicted_label)
            if predicted_label == key:
                correct_count += 1
        accuracy_scores[key] = correct_count / len(images)
    k_accuracy_scores[i]['accuracy'] = accuracy_scores
    score = np.mean(list(accuracy_scores), axis=0)
    print(f"Accuracy for k={i}: {score:.4f}")

image_matrix = train_images[0].reshape((16, 16))
plt.imshow(image_matrix, cmap='gray')
plt.colorbar(label="Value")

plt.show()