import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD DATASET
# =========================

dataset_path = "dataset"

faces = []
labels = []

for file in os.listdir(dataset_path):

    img_path = os.path.join(dataset_path, file)

    # baca gambar grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # resize gambar
    img = cv2.resize(img, (100, 100))

    # ubah jadi vector
    vector = img.flatten()

    faces.append(vector)

    labels.append(file)

faces = np.array(faces)

# =========================
# MENAMPILKAN VARIABEL
# =========================

print("Jumlah data:", len(faces))
print("Ukuran matrix:", faces.shape)

# =========================
# MEAN FACE
# =========================

mean_face = np.mean(faces, axis=0)

print("Ukuran mean face:", mean_face.shape)

# =========================
# NORMALISASI DATA
# =========================

A = faces - mean_face

print("Ukuran matrix A:", A.shape)

# =========================
# SVD
# =========================

U, S, VT = np.linalg.svd(A, full_matrices=False)

print("Ukuran U:", U.shape)
print("Ukuran S:", S.shape)
print("Ukuran VT:", VT.shape)

# =========================
# EIGENFACE
# =========================

eigenfaces = VT[:3]

print("Ukuran eigenfaces:", eigenfaces.shape)

# =========================
# TAMPILKAN EIGENFACE
# =========================

plt.figure(figsize=(15,5))

for i in range(3):

    plt.subplot(1, 3, i + 1)

    eigenface_img = eigenfaces[i].reshape(100, 100)

    plt.imshow(eigenface_img, cmap='gray')

    plt.title(f"EigenFace {i+1}")

    plt.axis('off')

plt.show()

# =========================
# INPUT TEST
# =========================

test_image = cv2.imread(
    os.path.join(dataset_path, "images2.png"),
    cv2.IMREAD_GRAYSCALE
)

test_image = cv2.resize(test_image, (100, 100))

test_vector = test_image.flatten()

# =========================
# HITUNG BOBOT WAJAH TEST
# =========================

weights = np.dot(
    eigenfaces,
    (test_vector - mean_face)
)

print("Bobot wajah:", weights)

# =========================
# FACE RECOGNITION
# =========================

min_dist = float('inf')
best_match = ""

for i in range(len(faces)):

    train_weight = np.dot(
        eigenfaces,
        (faces[i] - mean_face)
    )

    dist = np.linalg.norm(
        weights - train_weight
    )

    print(f"Distance dengan {labels[i]} :", dist)

    if dist < min_dist:
        min_dist = dist
        best_match = labels[i]

# =========================
# HASIL DETEKSI
# =========================

print("\n===== HASIL DETEKSI =====")
print("Wajah dikenali sebagai:", best_match)
print("Distance terkecil:", min_dist)

if min_dist < 5000:
    print("Wajah cocok")
else:
    print("Wajah tidak cocok")