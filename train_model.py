# train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ----------------------------
# 1. Load preprocessed data
# ----------------------------
df = pd.read_csv('news_clean.csv')

# ----------------------------
# 2. Clean data
# Drop rows with NaN or empty content/label
df = df.dropna(subset=['content', 'label'])
df = df[df['content'].str.strip() != '']

# Map labels to numeric: 0 = REAL, 1 = FAKE
df['label_num'] = df['label'].map({'REAL': 0, 'FAKE': 1})

# Drop rows where mapping failed (any remaining NaN in label_num)
df = df.dropna(subset=['label_num'])
df['label_num'] = df['label_num'].astype(int)

# ----------------------------
# 3. Prepare features and labels
# ----------------------------
X = df['content'].values
Y = df['label_num'].values

# ----------------------------
# 4. Convert text to TF-IDF vectors
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# ----------------------------
# 5. Split into training and test sets
# ----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# ----------------------------
# 6. Train Logistic Regression model
# ----------------------------
model = LogisticRegression()
model.fit(X_train, Y_train)

# ----------------------------
# 7. Evaluate model
# ----------------------------
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

print(f"Training Accuracy: {train_acc}")
print(f"Test Accuracy: {test_acc}")

# ----------------------------
# 8. Save model and vectorizer
# ----------------------------
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved successfully!")