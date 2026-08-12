import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# 1. Üç grup hâlinde yapay veri oluştur.
X, real_answers = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.70,
    random_state=42
)

print("X şekli:", X.shape)
print("İlk 5 veri:")
print(X[:5])


# real_answers değerlerini eğitimde kullanmayacağız.
# Çünkü denetimsiz öğrenme yapıyoruz.
print("\nGerçek cevaplar eğitimde kullanılmıyor.")


# 2. K-Means modelini oluştur.
kmeans = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=42
)


# 3. Merkezleri öğren ve her noktaya küme numarası ver.
cluster_labels = kmeans.fit_predict(X)

print("\nİlk 10 küme tahmini:")
print(cluster_labels[:10])


# 4. Modelin bulduğu merkezler.
centroids = kmeans.cluster_centers_

print("\nBulunan merkezler:")
print(centroids)


# 5. Yeni noktaların kümelerini tahmin et.
new_points = np.array([
    [-7, -7],
    [0, 2],
    [5, 2]
])

new_predictions = kmeans.predict(new_points)

print("\nYeni noktalar:")
print(new_points)

print("\nYeni noktaların küme tahminleri:")
print(new_predictions)


# 6. Yeni noktaların önceki eğime göre test ediyor.
distances = kmeans.transform(new_points)

print("\nYeni noktaların merkezlere uzaklıkları:")
print(np.round(distances, 2))


# 7. Sonucu çizdir.
plt.scatter(
    X[:, 0],          # Bütün noktaların x1 değerleri → yatay eksen Bütün satırları al, sadece 0. sütunu seç.
    X[:, 1],          # Bütün noktaların x2 değerleri → dikey eksen Bütün satırları al, sadece 1. sütunu seç.
    c=cluster_labels, # Küme numarasına göre renk
    cmap="viridis",   # Kullanılacak renk paleti
    alpha=0.70,       # Noktaların saydamlığı
    label="Veriler"   # Grafiğin açıklama kutusundaki isim
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    marker="X",
    s=250,
    label="Centroidler"
)

plt.scatter(
    new_points[:, 0],
    new_points[:, 1],
    c="black",
    marker="*",
    s=180,
    label="Yeni noktalar"
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("K-Means ile etiketsiz verileri gruplama")
plt.grid() #Arka plan çizgileri
plt.legend()  #Açıklama kutusu
plt.show()