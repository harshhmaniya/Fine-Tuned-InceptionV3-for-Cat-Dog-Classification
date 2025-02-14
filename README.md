# 🐱🐶 Fine-Tuned InceptionV3 & VGG19 for Cat-Dog Classification

This project demonstrates transfer learning for classifying cat and dog images using two pre-trained models: **InceptionV3** and **VGG19**. The provided notebook implements the complete pipeline—from dataset loading and preprocessing to model fine tuning, evaluation, and prediction.

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Setup & Usage](#-setup--usage)
3. [Dataset & Preprocessing](#-dataset--preprocessing)
4. [Model & Training](#-model--training)
5. [Evaluation](#-evaluation)
6. [Making Predictions](#-making-predictions)
7. [License](#-license)
8. [Connect with Me](#-connect-with-me)

---

## 🔍 Overview

- **Objective:** Classify images of cats and dogs.
- **Models:** Fine-tuning of InceptionV3 and (optionally) VGG19.
- **Notebook:** Run all cells in `cat_dog_class.ipynb` to reproduce the results.

---

## 🛠️ Setup & Usage

1. **Install Dependencies:**  
   ```bash
   pip install kagglehub matplotlib seaborn numpy tensorflow
   ```
2. **Run the Notebook:**  
   Open `cat_dog_class.ipynb` in Jupyter or Google Colab and execute cells sequentially.

3. **Dataset:**  
   The notebook downloads the [tongpython/cat-and-dog](https://www.kaggle.com/tongpython/cat-and-dog) dataset via kagglehub.

---

## 📂 Dataset & Preprocessing

- **Training Set:** 8005 images  
- **Test Set:** 2023 images  
- **Preprocessing:**  
  - Images are rescaled by **1/255**.  
  - **Training Augmentation:**  
    - Shear: 0.2  
    - Zoom: 0.2  
    - Horizontal Flip  
  - **Target Size:** 224×224  
  - **Batch Size:** 32

---

## 🧠 Model & Training

- **Base Models:**  
  - **InceptionV3:** Loaded with ImageNet weights, with all layers frozen except the last 10.  
  - **VGG19:** Imported for optional fine tuning (similar approach as InceptionV3).

- **Custom Top Layers:**  
  - **Architecture:** Flatten → Dense (512, ReLU) → Dense (256, ReLU) → Dense (1, Sigmoid)  
  - **Parameter Summary (InceptionV3):**  
    - **Total:** ~48M  
    - **Trainable:** ~26M  
    - **Non-trainable:** ~21M

- **Training Setup:**  
  - **Optimizer:** Adam (lr=0.001)  
  - **Loss Function:** Binary Crossentropy  
  - **Epochs:** 5  
  - **Batch Size:** 32

---

## ✅ Evaluation

- **InceptionV3:** Achieved a test accuracy of **~98.9%** (Test Loss: **~0.0446**).  
- **VGG19:** Imported for potential use; evaluation metrics for VGG19 are not separately provided.

---

## 🔮 Making Predictions

To use the trained model for predictions:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess an image
img = image.load_img('path_to_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict using the trained classifier
prediction = classifier.predict(img_array)

# Interpret the result: if prediction > 0.5, classify as dog; else, cat.
result = "dog" if prediction[0][0] > 0.5 else "cat"
print("Predicted Class:", result)
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Connect with Me

- [GitHub](https://github.com/harshhmaniya)  
- [LinkedIn](https://www.linkedin.com/in/harshhmaniya)  
- [Hugging Face](https://huggingface.co/harshhmaniya)
