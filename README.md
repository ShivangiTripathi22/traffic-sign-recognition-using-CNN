# 🚦 Traffic Sign Recognition using CNN

A deep learning-based **Traffic Sign Recognition system** that classifies traffic-sign images into **59 different categories** using a Convolutional Neural Network (CNN).

The project includes image preprocessing, CNN model training, performance evaluation, and an interactive **Streamlit web application** for testing the trained model on new traffic-sign images.

---

## 📌 Project Overview

Traffic sign recognition is an important computer vision task for intelligent transportation and driver-assistance systems.

In this project, a CNN was trained to recognize traffic signs from the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The trained model takes a traffic-sign image as input and predicts the most likely traffic-sign category along with its confidence score.

### Key Features

* 🖼️ Traffic-sign image classification
* 🧠 CNN-based deep learning model
* 🔢 Classification across 59 traffic-sign classes
* 📊 Model evaluation using accuracy, precision, recall, and F1-score
* 📈 Confusion matrix for class-level analysis
* 🎯 Top-5 prediction probabilities
* 🌐 Interactive Streamlit web application
* ⚡ Image-based prediction using the trained model

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Pandas**
* **Pillow**
* **OpenCV**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit**

---

## 📂 Dataset

The project uses the Indian-Traffic Sign-Dataset dataset.

The model performs multiclass classification across **59 traffic-sign categories**.

The images are preprocessed before being provided to the CNN.

### Image Preprocessing

Each input image is:

1. Converted to RGB
2. Resized to **32 × 32 pixels**
3. Converted into a NumPy array
4. Converted to `float32`
5. Normalized by dividing pixel values by `255`

This produces input data in the range:

The same preprocessing pipeline is used in the Streamlit application to maintain consistency with model training.

---

## 🧠 CNN Model

The project uses a Convolutional Neural Network designed for image classification.

The model learns visual features from traffic-sign images through convolutional and pooling layers before making the final classification.

The final model was saved in Keras format as:

```text
traffic_sign_cnn.keras
```

---

## 📊 Model Performance

The trained CNN was evaluated on the test dataset.

| Metric             |     Result |
| ------------------ | ---------: |
| Test Accuracy      | **82.79%** |
| Weighted Precision |   **0.84** |
| Weighted Recall    |   **0.84** |
| Weighted F1-Score  |   **0.84** |
| Number of Classes  |     **59** |
| Test Samples       |  **2,795** |

Performance varies across individ
