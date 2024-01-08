# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Random Forest modelini yükle
model_path = "C:\\Users\\goaud\\Desktop\\ml laptop\\RandomForestRegressor\\random_forest_regression_model.joblib"
model = joblib.load(model_path)

# Streamlit uygulama başlığı
st.title("Laptop Fiyat Tahmini")

# Kullanıcıdan giriş al
st.sidebar.header("Laptop Özellikleri")
st.sidebar.markdown("Lütfen aşağıdaki özelliklere uygun değerleri girin:")

# Veri setini yükle
dataset_path = "C:\\Users\\goaud\\Desktop\\ml laptop\\Preprocessing\\laptop-fill-training.xlsx"
dataset = pd.read_excel(dataset_path)

# Marka seçimi
brands = dataset["Marka"].unique()  # Veri setindeki benzersiz marka isimlerini alın
selected_brand = st.sidebar.selectbox("Marka", brands)

# Diğer özellikleri girme
feature_names = ["Ekran Boyutu", "Çözünürlük", "Ekran Yenileme Hızı", "İşlemci Tipi", "İşlemci Çekirdek Sayısı", "Ram (Sistem Belleği)", "SSD Kapasitesi", "Ekran Kartı", "Ekran Kartı Hafızası", "İşletim Sistemi", "Cihaz Ağırlığı"]
features = []
for feature_name in feature_names:
    feature = st.sidebar.number_input(f"{feature_name.capitalize()}", value=0.0, step=0.1)
    features.append(feature)

# Tahmin butonu
if st.sidebar.button("Fiyatı Tahmin Et"):
    # Seçilen marka değerini kullanıcının girdiği özelliklere ekleyin
    selected_brand_row = dataset[dataset["Marka"] == selected_brand].iloc[0]
    input_features = np.array([selected_brand_row["Marka"]] + features).reshape(1, -1)
    
    # Kullanıcının girdiği özellikleri kullanarak fiyat tahmini yap
    predicted_price = model.predict(input_features)

    # Tahmin sonucunu göster
    st.success(f"Tahmini Fiyat: {predicted_price[0]:.2f} TL")
