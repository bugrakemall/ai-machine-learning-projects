from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist.data, mnist.target

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

rng = np.random.default_rng(seed=42)
noise_train = rng.integers(0,100,(len(X_train),784))

X_train_mod = X_train + noise_train

noise_test = rng.integers(0,100,(len(X_test),784))

X_test_mod = X_test + noise_test

y_train_mod = X_train #Temiz resimler

y_test_mod = X_test #Temiz resimler

knn_clf = KNeighborsClassifier() #benzerlerine bakarak tahmin yapan model, yeni örnek gelince benzerlerine bakar.

knn_clf.fit(X_train_mod,y_train_mod) #Kirli resmi al, böyle temizle diye eğitiyoruz

clean_digit = knn_clf.predict([X_test_mod[1]])

plot_digit(clean_digit)
plt.show()

