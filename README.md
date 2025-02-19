# 📩 Spam Email Prediction

## 📌 Overview
This project focuses on **Spam Email Detection** using **Machine Learning (Logistic Regression)**. The model is trained on a dataset of email messages and predicts whether an email is **Spam (0)** or **Ham (1)**.

## 📚 Dataset
- The dataset is loaded from a CSV file (`mail_data.csv`).
- It contains two columns:
  - **Category** (Spam or Ham)
  - **Message** (Email text)
- Missing values are replaced with empty strings.

## 🛠️ Tech Stack
- **Programming Language**: Python 🐍
- **Libraries Used**:
  - `numpy`
  - `pandas`
  - `scikit-learn`

## 🛠️ Model Development

### 1️⃣ Importing Required Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### 2️⃣ Loading the Dataset
```python
raw_mail_data = pd.read_csv('mail_data.csv')
print(raw_mail_data.head())

# Handling missing values
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
```

### 3️⃣ Data Preprocessing
```python
# Converting labels to numerical values
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

X = mail_data['Message']
Y = mail_data['Category'].astype(int)  # Convert labels to integers
```

### 4️⃣ Splitting the Data
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
```

### 5️⃣ Feature Extraction using TF-IDF
```python
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
```

### 6️⃣ Model Training
```python
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```

### 7️⃣ Model Evaluation
```python
# Training Accuracy
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data:', accuracy_on_training_data)

# Testing Accuracy
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data:', accuracy_on_test_data)
```

### 8️⃣ Making Predictions
```python
input_mail = ["I've been searching for the right words to thank you for this breather."]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')
```

## 🚀 Results
- **Training Accuracy**: ✅ *Displayed in the output*
- **Testing Accuracy**: ✅ *Displayed in the output*
- The model successfully classifies emails as **Spam** or **Ham**.

## 📌 How to Run the Project
1. Clone this repository:
   ```sh
   git clone https://github.com/Bhupathirayudu567/Spam_Email_Prediction.git
   cd Spam_Email_Prediction
   ```
2. Install dependencies:
   ```sh
   pip install numpy pandas scikit-learn
   ```
3. Run the script:
   ```sh
   python spam_email_detection.py
   ```

## 🌟 Acknowledgment
This project was developed using **Scikit-Learn** and standard **ML techniques**.

