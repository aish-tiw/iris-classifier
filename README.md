# Iris Classifier 

## Overview

This project builds a machine learning model to classify iris flowers into one of three species:
- Setosa
- Versicolor
- Virginica

We use two models:
- Decision Tree Classifier (with hyperparameter tuning)
- K-Nearest Neighbors (KNN)

Evaluation includes accuracy, confusion matrix and feature importance. The notebook was developed in Google Colab using the classic Iris dataset from scikit-learn.

---

Click below to open the notebook directly in Colab — no installation needed:

[![Open In Colab](https://colab.research.google.com/github/aish-tiw/iris-classifier/blob/main/notebooks/iris_model.ipynb
)

---

To run this locally, use these steps:

```bash
git clone https://github.com/aish-tiw/iris-classifier.git
cd iris-classifier
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/iris_model.ipynb



