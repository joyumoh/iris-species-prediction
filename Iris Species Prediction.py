#!/usr/bin/env python
# coding: utf-8

# Install and import libraries

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_recall_curve



# Loading dataset

# In[2]:


df = pd.read_csv('Iris.csv')
df.head(10)


# Understanding the data

# In[3]:


df.info()
df.describe()


# Data cleaning

# In[4]:


## checking for duplicates
print('== Duplicate Values ==')
print (df.duplicated().sum())

## checking for missing values
print('== Missing Values ==')
df.isnull().sum()


# EDA

# In[5]:


## Distribution of data
plt.figure(figsize=(8,4))
sns.countplot(x = 'Species', data = df)
plt.title('Species Distribution')
plt.show()
print('The flower species are equally distributed')


# In[6]:


## Correlation of numerical features
sns.pairplot(df, hue='Species', height=1.1, aspect=1.5, palette='deep', diag_kind='kde')
plt.show()
print('Interpretation: Iris-setosa shows a distinct characteristic from the others as shown in the clusters formed, Petal length and width is most effective in distinguishing species. Strong correlation between petal length and width, as one increases the other also increases. There are some overlap in the measurement of some species especially with Sepal length and width, making it difficult to distinguish them using only sepal measurements.')


# In[7]:


## Distribution of the data
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

# Define colors for each feature plot
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
titles = [
    "Sepal Length Distribution",
    "Sepal Width Distribution",
    "Petal Length Distribution",
    "Petal Width Distribution",
]

# Loop through features and plot histograms
for i, ax in enumerate(axes.flat):
    sns.histplot(
        data=df,
        x=features[i],
        color=colors[i],
        kde=True,
        ax=ax,
        edgecolor="black",
    )
    ax.set_title(titles[i], fontsize=12, fontweight="bold")
    ax.set_xlabel("Measurement (cm)")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.show()
print('Interpretation:')
print('Sepal Length Distribution: This shows a relatively symmetric and bell-shaped normal distribution, most of the flowers have sepal length between 5.0cm and 6.5cm, with the peak at 5.8cm.')
print('Sepal Width Distribution: This shows a strong narrow normal distribution, they cluster tightly around the center, with majority between 2.8 and 3.4cm, with a clear distinct peak at 3cm.')

print('Petal Length Distribution: This shows a bimodal distribution, one sharp isolated peak on the left around 1.5cm and a broader peak on the right between 4.0 and 5.5cm. This gap exists because one specie-Iris setosa has much smaller petals than others.')
print('Petal Width Distribution: Same as the petal length, one peak on the left between 0.2 and 0.4cm and the second wider peaks around 1.3 and 1.8cm.')


# Feature Selection

# In[8]:


x=df.drop(['Id', 'Species'], axis=1)
y= df['Species']


# Train-Test Split

# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, stratify=y, random_state=42)
print('Features shape:', x_train.shape)
print('Labels shape:', y_train.shape)
print(y_train.value_counts())


# Models Training

# In[38]:


# Training LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)


# In[39]:


# Training RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)


# Predictions

# In[11]:


# Predicting using logisticregression
y_pred_lr = lr.predict(x_test)

# Predicting using randomforestclassifier
y_pred_rf = rf.predict(x_test)

# Displaying predictions and actual
pred_lr_df = pd.DataFrame(y_pred_lr, columns = ['lr_Predicted'])
pred_rf_df = pd.DataFrame(y_pred_rf, columns = ['rf_Predicted'])
y_test_df = pd.DataFrame(y_test).reset_index(drop = True)
results = pd.concat([y_test_df, pred_lr_df, pred_rf_df], axis = 1)
print("Model Predictions vs Actual:")
display(results.head(10))


# Models Validation

# In[17]:


# Configure and run cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
score1 = cross_val_score(lr, x_train, y_train, cv=kfold, scoring='accuracy')
score2 = cross_val_score(rf, x_train, y_train, cv=kfold, scoring='accuracy')

print(f"Mean Accuracy1: {np.mean(score1):.4f}")
print(f"Mean Accuracy2: {np.mean(score2):.4f}")
print(f"LogisticRegression model achieved a slightly better performance than RandomForestClassifier")


# Models Evaluation

# In[21]:


y_true=y_test
y_pred_lr = lr.predict(x_test)
y_pred_rf = rf.predict(x_test)

# LogisticRegression Evaluation
accuracy_lr=accuracy_score(y_true, y_pred_lr)
class_report_lr=classification_report(y_true, y_pred_lr)
conf_matrix_lr=confusion_matrix(y_true,y_pred_lr)

print('=== LogisticRegression ===')
print(f'Overall Model Accuracy: {accuracy_lr * 100:.2f}%')
print(f'Classification Report: \n{class_report_lr}')
print(f'Confusion Matrix: \n{conf_matrix_lr}')


# RandomForestClassifier Evaluation
accuracy_rf=accuracy_score(y_true, y_pred_rf)
class_report_rf=classification_report(y_true, y_pred_rf)
conf_matrix_rf=confusion_matrix(y_true,y_pred_rf)

print('=== RandomForestClassifier ===')
print(f'Overall Model Accuracy: {accuracy_rf * 100:.2f}%')
print(f'Classification Report: \n{class_report_rf}')
print(f'Confusion Matrix: \n{conf_matrix_rf}')


# In[37]:


# Visualize the confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(conf_matrix_lr, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True, ax=axes[0])
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")
axes[0].set_title('LR Confusion Matrix')

sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Greens", xticklabels=True, yticklabels=True)
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")
axes[1].set_title('RF Confusion Matrix')

# Displlay the figures
plt.tight_layout()
plt.show()

print('LR Confusion Matrix: The model correctly classified Iris-setosa species, misclassified 1 instance of Iris-versicolor as Iris-virginica, also misclassified 2 instances of Iris-virginica as Iris-versicolor.')
print('RF Confusion Matrix: The model correctly classified Iris-setosa species, misclassified 1 instance of Iris-versicolor as Iris-virginica, also misclassified 4 instances of Iris-virginica as Iris-versicolor. ')
print('The confusion matrix shows that LogisticRegression has a better performance than RandomForestClassifier.')

