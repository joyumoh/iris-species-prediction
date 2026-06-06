# Iris Species Prediction

A machine learning project that classifies Iris flowers into three distinct species (Setosa, Versicolor, and Virginica) using physical measurements, data visualization, and predictive modeling.

---

## 📋 Table of Contents
* [Project Overview](#-project-overview)
* [Project Structure](#-project-structure)
* [Dataset Information](#-dataset-information)
* [Setup Instructions](#-setup-instructions)
* [How to Run](#-how-to-run)
* [Key Features & Visualizations](#-key-features--visualizations)

---

## 🔍 Project Overview
This repository showcases an end-to-end data science pipeline. It takes the classic Iris flower dataset, performs Exploratory Data Analysis (EDA) with clear visual plots, and provides Python scripts to train a model and make real-time predictions.

## 📁 Project Structure
The repository contains the following core files:
* **`Iris Species Prediction.ipynb`**: Jupyter Notebook containing full data exploration, plotting, and model experimentation.
* **`Iris Species Prediction.py`**: Clean, production-ready Python script exported directly from the notebook pipeline.
* **`main.py`**: The main interface/entry point script to easily run calculations and check new predictions.
* **`iris.csv`**: The clean source dataset containing target features and species labels.
* **`requirements.txt`**: The comprehensive dependency manifest tracking required library versions.
* **`.gitignore`**: Configuration file specifying local untracked folders (`.venv/`, `.idea/`) to keep the repository tidy.

## 📊 Dataset Information
The dataset tracks 4 distinct physical attributes across 150 instances of Iris flowers:
1. **Sepal Length** (in cm)
2. **Sepal Width** (in cm)
3. **Petal Length** (in cm)
4. **Petal Width** (in cm)

**Target Labels:** `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone [https://github.com](https://github.com/joyumoh/iris-species-prediction)
```
*(Make sure to replace the placeholder link with your actual GitHub project link after creating it).*

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Ensure your package tools are updated and install your specific requirement tracking profile:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Execute the Core Pipeline
To execute the primary classification logic and preview model outcomes directly from your terminal console, run:
```bash
python main.py
```

---

## 📈 Key Features & Visualizations
* **Clustering Analysis**: The project isolates and tracks petal dimensions which easily group and differentiate species like *Iris-Setosa*.
* **Pairplots & Distribution Charts**: Built-in script generation yields distribution profiles to clearly demonstrate feature overlap between *Versicolor* and *Virginica*.
* **Clean Code Separation**: Keeps exploratory analysis separate from clean, operational Python files.
*
