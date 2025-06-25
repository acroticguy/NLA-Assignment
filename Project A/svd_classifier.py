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

svd_map = {key: {} for key in range(10)}

for label, images in train_map.items():
    U, S, Vt = np.linalg.svd(images.T, full_matrices=False)
    
    svd_map[label]['vectors'] = U
    svd_map[label]['s_values'] = S

k_accuracy_scores = {i: {} for i in range(20, 4, -1)}

for i in range(20, 4, -1):
    accuracy_scores = [0 for _ in range(10)]
    test_pred = {key: [] for key in range(10)}
    total_correct = 0
    for label, images in test_map.items():
        correct_count = 0
        for image in images:
            residual_norm = []
            for j in range(10):
                projection = svd_map[j]['vectors'][:, :i] @ svd_map[j]['vectors'][:, :i].T @ image
                residual_norm.append(np.linalg.norm(image - projection))
            residual_norm = np.array(residual_norm)
            predicted_label = np.argmin(residual_norm)
            test_pred[label].append(predicted_label)
            if predicted_label == label:
                correct_count += 1
        accuracy_scores[label] = correct_count / len(images)
        total_correct += correct_count
    k_accuracy_scores[i]['accuracy_macro'] = accuracy_scores
    score = total_correct / test_images.shape[0]
    k_accuracy_scores[i]['accuracy'] = score
    print(f"Accuracy for k={i}: {score:.4f}")
max_k = max(k_accuracy_scores, key=lambda k: k_accuracy_scores[k]['accuracy'])
print(f"Best k value: {max_k} with accuracy {k_accuracy_scores[max_k]['accuracy']:.4f}")

confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
test_pred = {key: [] for key in range(10)}

for label, images in test_map.items():
    correct_count = 0
    for image in images:
        residual_norm = []
        for j in range(10):
            projection = svd_map[j]['vectors'][:, :max_k] @ svd_map[j]['vectors'][:, :max_k].T @ image
            residual_norm.append(np.linalg.norm(image - projection))
        residual_norm = np.array(residual_norm)
        predicted_label = np.argmin(residual_norm)
        confusion_matrix[predicted_label][label] += 1
        test_pred[label].append(predicted_label)
        if predicted_label == label:
            correct_count += 1

confusion_matrix = np.array(confusion_matrix)
print(f"Confusion matrix: \n{confusion_matrix}")

cm_no_diag = confusion_matrix.copy()
np.fill_diagonal(cm_no_diag, 0)

max_error_idx = np.argmax(cm_no_diag)

true_label, pred_label = np.unravel_index(max_error_idx, cm_no_diag.shape)

print(f"The most difficult digit is {true_label}, often mistaken for {pred_label}. It was confused {cm_no_diag[true_label, pred_label]} times.")

example_i = test_pred[true_label].index(pred_label)

example = test_map[true_label][example_i]

image_matrix = example.reshape((16, 16))
plt.imshow(image_matrix, cmap='gray')
plt.colorbar(label="Value")

plt.show()

plt.figure(figsize=(12, 7))
for label, val in svd_map.items():
    # Plot the first 30 singular values for each class
    plt.plot(range(1, len(val['s_values'][:30]) + 1), val['s_values'][:30], marker='.', linestyle='-', label=f'Class {label}')

plt.title('Singular Values for Each Digit Class')
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value Magnitude (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
