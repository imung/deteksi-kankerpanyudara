
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("breast-cancer.csv")

# Label encoding diagnosis
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

# Gunakan hanya 10 fitur utama
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]
y = df['diagnosis']

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_bc.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Latih model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "model_bc_randomforest.pkl")

print("✅ Model dan scaler berhasil disimpan dengan 10 fitur.")
