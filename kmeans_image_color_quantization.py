import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.cluster import KMeans

filepath = "ladybug.png"


#pikselin kesin olarak 3 değerden oluşmasını sağlar.
image = Image.open(filepath).convert("RGB")

#Resmi Numpy arraya dönüştürür.
image = np.asarray(image)

#(height,weight,3-RGB values-)
#print("Orijinal görüntünün şekli:", image.shape)

#Kmeans için resmi uzun bir piksel listesine dönüştürüyoruz.
# -1: Satır sayısını NumPy kendisi hesaplasın.
#  3: Her satırda R, G ve B olmak üzere 3 değer bulunsun.
X= image.reshape(-1,3)

#print("Piksel listesinin şekli:", X.shape)


#Burada KMeansin görevi binlerce farklı RGB rengini 8 benzer renk grubuna ayırmaktır.
kmeans = KMeans(
    n_clusters=8, #Resimdeki bütün renkleri 8 renk kümesine ayır.
    random_state=42, 
    n_init=10 #10 farklı centroid başlangıcı dene, inertia'sı en düşük sonucu sakla
)
# Modeli bütün piksel renkleriyle eğit.
# 1. Benzer RGB renklerini aynı kümeye koyar. # 2. Her renk kümesinin ortalama rengini hesaplar.# 3. Her piksele bir küme numarası verir.
kmeans.fit(X)

#K-Means’in bulduğu 8 ortalama rengi verir. 
#Bu renkler, kümelerdeki RGB değerlerinin ortalaması alınarak hesaplanır.
centroids = kmeans.cluster_centers_
#print(centroids)

#labels_: Her pikselin hangi merkeze, yani hangi renk kümesine en yakın olduğunu gösterir.
pixel_labels = kmeans.labels_
print(pixel_labels)
#Resmin şekli (533,800,3) 533*800 = 426400 tane pixel var her pixel 0-7 arasında etiketleniyor çünkü n_cluster=8 dir.

# Her pikseli, ait olduğu kümenin merkez rengiyle değiştir.
#
# Örneğin pixel_labels[0] = 3 ise:
# İlk pikselin yeni rengi centroids[3] olur.
#
# Böylece bütün pikseller artık yalnızca
# K-Means'in bulduğu 8 renkten birini kullanır.
segmented_img = centroids[pixel_labels]


# Şu anda görüntü hâlâ uzun bir piksel listesidir:
# (426400, 3)
#
# Bunu tekrar orijinal resmin şekline çevir:
# (533, 800, 3)
segmented_img = segmented_img.reshape(image.shape)

# Renk değerlerini 0-255 arasında tam sayılara dönüştür.
segmented_img = np.clip(segmented_img, 0, 255).astype(np.uint8)

plt.figure(figsize=(12,5)) #12cm yatay 5cm dikey

# 1 satır ve 2 sütundan oluşan alanın 1. bölümünü seçer.
plt.subplot(1, 2, 1)

# Orijinal resmi seçilen bölüme çizer.
plt.imshow(image)

plt.imshow(segmented_img)

plt.show()

#KMeans'in amacı cevabı olmayan sayısal verilerde birbirine benzeyen örnekleri gruplandırmaktır.
#Ne zaman kullanılır? Elinde çok fazla veri var ama sınıf cevapları yoksa benzerleri gruplamak için.