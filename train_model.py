import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
import pickle
import os

print("Loading dataset...")
df = pd.read_csv('cleaned_dataset_Thyroid1.csv')
print(df.head())
print(df.info())

print("\nHandling missing values...")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

imputer_num = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

if categorical_cols:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

print(df.isnull().sum())

print("\nEncoding categorical variables...")
if 'binaryClass' in df.columns:
    df['binaryClass'] = df['binaryClass'].map({0: 'Thyroid Negative', 1: 'Thyroid Positive'})

# Separate features and target BEFORE extracting column names
X = df.drop('binaryClass', axis=1)
y = df['binaryClass']

# NOW get the feature column names (without binaryClass)
numerical_cols = [col for col in numerical_cols if col != 'binaryClass']
categorical_cols = [col for col in categorical_cols if col != 'binaryClass']

print(f"\nFeature columns: {len(X.columns)}")

le = LabelEncoder()
y = le.fit_transform(y)

# Use only numerical columns for scaling
if categorical_cols:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', 'passthrough', categorical_cols)
        ])
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols)
        ])

X_processed = preprocessor.fit_transform(X)

print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print("\nTraining Random Forest model...")
best_model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {acc:.4f}')
print(classification_report(y_test, y_pred))

print("\nSaving model files...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
print("✓ best_model.pkl saved")

with open(os.path.join(BASE_DIR, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)
print("✓ preprocessor.pkl saved")

with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)
print("✓ label_encoder.pkl saved")

print("\nAll files generated successfully!")