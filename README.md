# 👕🥿 Fashion MNIST Classifier (TensorFlow + TFLite)

This project trains a neural network on the Fashion MNIST dataset using **TensorFlow 2.x**, and converts it to a **TensorFlow Lite model (`.tflite`)** for mobile  deployment. It includes both the training pipeline and model export for real-world usage.

---

## ✅ Project Highlights

- 👚 Trains a model to classify clothing images
- 💾 Saves the trained model as `.h5` (Keras format)
- 📱 Converts the model to `.tflite` format optimized for mobile devices
- 🧠 Built using `Sequential` model with Dense & Dropout layers
- 📈 Achieves highest test accuracy on clean data

---

## 🧠 Techniques Used

- 🔧 **Sequential Keras model** with:
  - `Dense(128)` + `ReLU`
  - `Dropout(0.2)` for regularization
  - `Dense(10)` + `softmax` for 10-class output
- 🧼 **Data normalization**: scaling input pixels to `[0, 1]`
- 🔁 Trained for **5 epochs** with **Adam optimizer**
- ✅ Converted to `.tflite` using `TFLiteConverter` from Keras model

---

## ⚠️ Challenges Faced

### 🔄 TensorFlow Version Compatibility




