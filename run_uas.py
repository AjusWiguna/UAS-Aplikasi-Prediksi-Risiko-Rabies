import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_excel("C:/Users/ASUS TUF/Downloads/LAPORAN GHPR  DAN RABIES 2025 per bulan.xlsx", header=None)

print("Full data shape:", df.shape)
print(df.head(10))
print(df.tail(10))


df = pd.read_excel("C:/Users/ASUS TUF/Downloads/LAPORAN GHPR  DAN RABIES 2025 per bulan.xlsx", skiprows=9)


df.columns = ['No', 'PUSKESMAS', 'JUMLAH KASUS GHPR', 'KATEGORI III', 'JUMLAH VAR', 'JUMLAH SAR', 'GIGITAN ANJING', 'LYSSA / RABIES POSITIF'] + [f'col{i}' for i in range(8, len(df.columns))]

print("After setting column names:")
print(df.head())
print(df.info())
print(df.columns)

features = [
    "JUMLAH KASUS GHPR",
    "KATEGORI III",
    "JUMLAH VAR",
    "JUMLAH SAR",
    "GIGITAN ANJING"
]

target = "LYSSA / RABIES POSITIF"

df = df[features + [target]]
print(df.head())

# Convert to numeric, coerce errors to NaN
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN in target or features
df = df.dropna(subset=features + [target])

# Jika rabies positif > 0 → berisiko (1), jika tidak → 0
df[target] = df[target].apply(lambda x: 1 if x > 0 else 0)

print(df[target].value_counts())
print(df.isnull().sum())

# Isi nilai kosong dengan 0 (aman untuk data kasus)
df = df.fillna(0)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

def prediksi_rabies():
    print("=== Aplikasi Prediksi Risiko Rabies ===")

    ghpr = float(input("Jumlah Kasus GHPR: "))
    kat3 = float(input("Jumlah Gigitan Kategori III: "))
    var = float(input("Jumlah Vaksin VAR: "))
    sar = float(input("Jumlah Serum SAR: "))
    anjing = float(input("Jumlah Gigitan Anjing : "))

    data_input = np.array([[ghpr, kat3, var, sar, anjing]])
    data_input_scaled = scaler.transform(data_input)

    hasil = model.predict(data_input_scaled)

    if hasil[0] == 1:
        print("Risiko Rabies: TINGGI")
    else:
        print("Risiko Rabies: RENDAH")

# Jalankan fungsi prediksi
prediksi_rabies()
