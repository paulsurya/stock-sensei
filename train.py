import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from model import load_data, features

DATA_PATH = os.path.join(os.path.dirname(__file__), 'cache', 'historical', 'RELIANCE_NS_daily.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'final_model.pkl')

df = load_data(DATA_PATH)

print(f"Loaded {len(df)} rows, features: {features}")

x = df[features]
y = df['Target']

print(f"Class balance:\n{y.value_counts()}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

model = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'])}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
