from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

#multilabel: bir veri ayni anda birden fazla cevabi olabilir. Her etiket icin ayri noron vardir.
#binary class: sadece iki cevaptan biri, outputta sadece bir nöron vardır.
#multiclass: bir cok sinif vardir ama sadece birini icerebilir
#softmax 10 tane ciktiyi toplami 1 olan olasiliga cevirir.

fashion_mnist = fetch_openml(
    name="Fashion-MNIST",
    version=1,
    as_frame=False #Pandas yerine numPy array kullan.
)

X = fashion_mnist.data #kiyafet resimlerinin pixel değerleri

y= fashion_mnist.target.astype(int)

X_train = X[:60_000]
y_train = y[:60_000]

X_test = X[60_000:] #son 10k
y_test = y[60_000:]

# Model kıyafet isimlerini değil, 0-9 arasındaki sınıf numaralarını öğrenir.
#
# Bu liste, sınıf numarasını okunabilir isme çevirmemizi sağlar.
class_names = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot"    # 9
]

#X_train[0] 784 tane sayi içeren bir yatay arraydir.

X_sample = X_train[0].reshape(28,28) #bunu insan gözüyle görülebilecek resme çeviriyor.

correct_class_name = class_names[y_train[0]] 

plt.imshow(X_sample, cmap="binary") #cmap=binary görüntüde piksel değerleri 0 ile 255 arasındaysa, binary bu değerleri gri/siyah-beyaz tonlara çevirerek görselleştirir.

# Resmin üzerine gerçek sınıf adını yazar.
plt.title(correct_class_name)

# X ve Y eksen sayılarını gizler.
plt.axis("off")

# Hazırlanan resmi ekranda gösterir.
plt.show()

mlp_clf = MLPClassifier(
    hidden_layer_sizes=[300, 100],
    verbose=True, #Eğitim bilgilerini terminale yazdırır.
    early_stopping=True, #Bu, gereksiz eğitimi ve overfitting'i azaltır.
    random_state=42
)

pipeline = make_pipeline(
    MinMaxScaler(), # MinMaxScaler bunları 0-1 arasına getirir:
    mlp_clf
)

pipeline.fit(X_train,y_train) #mlp_clf pipelinein içinde olduğu için pipelineyi eğittik.

accuracy = pipeline.score(X_test, y_test) # score(), MLPClassifier için accuracy döndürür.

