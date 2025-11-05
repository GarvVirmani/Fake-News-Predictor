# train_model.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# ----------------------------
# 1. Load preprocessed data
# ----------------------------
df = pd.read_csv('news_clean.csv')

# ----------------------------
# 2. Clean data
# ----------------------------
df = df.dropna(subset=['content', 'label'])
df = df[df['content'].str.strip() != '']

# Map labels to numeric
df['label_num'] = df['label'].map({'REAL': 0, 'FAKE': 1})
df = df.dropna(subset=['label_num'])
df['label_num'] = df['label_num'].astype(int)

# ----------------------------
# 3. Prepare features and labels
# ----------------------------
X = df['content'].values
Y = df['label_num'].values

# ----------------------------
# 4. TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# ----------------------------
# 5. Train-test split
# ----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# ----------------------------
# 6. Define multiple models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(C=5.0, solver='liblinear', max_iter=2000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(C=0.5),
    "Decision Tree": DecisionTreeClassifier(random_state=2, max_depth=20),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=2)
}

train_accuracies = []
test_accuracies = []

# ----------------------------
# 7. Train, evaluate, compare
# ----------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, Y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f"{name} → Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# ----------------------------
# 8. Compare results visually
# ----------------------------
plt.figure(figsize=(10, 6))
bar_width = 0.35
x_labels = list(models.keys())
x = range(len(models))

plt.bar(x, train_accuracies, width=bar_width, label='Train Accuracy', alpha=0.7)
plt.bar([i + bar_width for i in x], test_accuracies, width=bar_width, label='Test Accuracy', alpha=0.7)

plt.xticks([i + bar_width/2 for i in x], x_labels, rotation=15)
plt.ylabel("Accuracy")
plt.title("Model Comparison on Fake News Classification")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 9. Save the best model
# ----------------------------
best_index = test_accuracies.index(max(test_accuracies))
best_model_name = x_labels[best_index]
best_model = models[best_model_name]

joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print(f"\n✅ Best Model: {best_model_name} (Test Accuracy: {max(test_accuracies):.4f}) saved successfully!")
