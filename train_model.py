import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

# Mengambil data dari file lokal
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Data awal: {df.shape}")

# Ubah TotalCharges ke angka (menangani string kosong " ")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


selected_features = [
    'tenure',            # Lama berlangganan
    'MonthlyCharges',    # Tagihan bulanan
    'TotalCharges',      # Total tagihan
    'Contract',          # Jenis Kontrak
    'InternetService',   # Jenis Internet
    'OnlineSecurity',    # Layanan Keamanan
    'TechSupport',       # Layanan Support
    'Churn'              # Target
]

df = df[selected_features]

# Kita gunakan MAPPING MANUAL agar konsisten dengan App.py nanti

# Mapping Manual
mapping_contract = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
mapping_internet = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
mapping_yes_no   = {'No': 0, 'Yes': 1, 'No internet service': 2}

# Terapkan Mapping
df['Contract'] = df['Contract'].map(mapping_contract)
df['InternetService'] = df['InternetService'].map(mapping_internet)
df['OnlineSecurity'] = df['OnlineSecurity'].map(mapping_yes_no)
df['TechSupport'] = df['TechSupport'].map(mapping_yes_no)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Pastikan tidak ada data kosong setelah mapping
df = df.dropna()

print("\nData setelah Preprocessing & Seleksi:")
print(df.head())

X = df.drop('Churn', axis=1)
y = df['Churn']

# Terapkan SMOTE agar data seimbang
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"\nJumlah data setelah SMOTE: {X_res.shape} (Seimbang)")

# Split Train & Test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Kita langsung gunakan Random Forest karena terbukti handal untuk data ini
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("\n=== HASIL EVALUASI RANDOM FOREST ===")
print("Akurasi  :", accuracy_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, 'telco_churn_model.pkl')
print("\nModel berhasil disimpan sebagai 'telco_churn_model.pkl'")
