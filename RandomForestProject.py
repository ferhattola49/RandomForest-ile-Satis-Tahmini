import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# CSV dosyasını yükle (Kendi dosya yolunuzu buraya ekleyin)
df = pd.read_csv('C:/Users/Administrator/Desktop/net_satis_verileri.csv')

# 'Gün' sütununu tarih formatına çevir (eğer tarih formatında değilse)
df['Gün'] = pd.to_datetime(df['Gün'])

# Özellik ve hedef değişkenleri ayır
X = df[['Gün']]
y = df['Net Satış Miktarı_MA']

# Tarihleri sayısal değerlere çevir
X['Gün'] = X['Gün'].map(pd.Timestamp.toordinal)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluştur ve eğit
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yap ve performansı değerlendir
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')

# Gelecek tarih aralığını ayarlama (2023-01-01'den 2025-01-01'e kadar)
future_dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='D')
future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal)

# Gelecek tarihler için tahmin yap
future_predictions = model.predict(future_dates_ordinal.values.reshape(-1, 1))

# Sonuçları ekrana yazdır
future_results = pd.DataFrame({
    'Tarih': future_dates,
    'Tahmin Edilen Net Satış Miktarı_MA': future_predictions
})

print("\nGelecek 2025'e Kadar Tahminler:")
print(future_results)

# 2023-04-10 tarihinden önceki verileri filtrele
filtered_df = df[df['Gün'] < '2023-04-10']

# Grafik oluşturma
plt.figure(figsize=(10, 5))

# 2023-04-10 tarihinden önceki gerçek değerler için çizgi grafiği
plt.plot(filtered_df['Gün'], filtered_df['Net Satış Miktarı_MA'], label='Gerçek Değerler', color='blue', linestyle='-', linewidth=2)

# Gelecek tahminler için çizgi grafiği
plt.plot(future_dates, future_predictions, label='Tahminler', color='orange', linestyle='-', linewidth=2)

# Başlık ve eksenler
plt.xlabel('Gün', fontsize=12)
plt.ylabel('Net Satış Miktarı', fontsize=12)
plt.title('Gerçek ve Tahmin Edilen Net Satış Miktarları', fontsize=14)

# Legend ve görünüm ayarları
plt.legend()

# Grafik görünümünü düzenle
plt.tight_layout()

# Grafik gösterimi
plt.show()
