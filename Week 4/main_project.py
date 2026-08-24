import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             confusion_matrix, ConfusionMatrixDisplay, 
                             roc_curve, auc)
import matplotlib.pyplot as plt

pd.set_option('future.no_silent_downcasting', True)

data = pd.read_csv('heart_disease_uci.csv')
data = data.dropna(subset=['thal'])
X = data.drop(['id', 'dataset', 'thal'], axis=1)
y = data['thal'].values

for col in X.columns:
    if X[col].dtype == 'object':
        if not X[col].mode().empty:
            X[col] = X[col].fillna(X[col].mode()[0])
    else:
        X[col] = X[col].fillna(X[col].median())


le = LabelEncoder()
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
X['sex'] = le.fit_transform(X['sex'])
X['fbs'] = le.fit_transform(X['fbs'])
X['exang'] = le.fit_transform(X['exang'])
X['exang'] = le.fit_transform(X['exang'])

for col in ['cp', 'restecg', 'slope']:
    encoded_array = ohe.fit_transform(X[[col]])
    
    encoded_col_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
    
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names, index=X.index)
    X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)

lr = LogisticRegression(max_iter=10000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = lr.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"accuracy score: {accuracy:.4f}")
print(f"precision score: {precision:.4f}")
print(f"recall score: {recall:.4f}")
print(f"confusion matrix: \n{cm}")

print(f"predictions: \n{y_pred}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Confusion Matrix')

classes = model.classes_
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

axes[1].plot([0, 1], [0, 1], 'k--', lw=2)
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].legend(loc="lower right")
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()