import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# ============================================================
# 1. VERİYİ YÜKLE
# ============================================================

housing = fetch_california_housing()

# X: Evlerin 8 özelliği
# y: Evlerin gerçek fiyatları
X = housing.data
y = housing.target


# ============================================================
# 2. TRAIN, VALIDATION VE TEST OLARAK AYIR
# ============================================================

# Bütün verinin %20'sini test için ayır.
X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Kalan %80'in %20'sini validation için ayır.
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid,
    y_train_valid,
    test_size=0.20,
    random_state=42
)


# Eşleşmeler:
#
# X_train ↔ y_train
# X_valid ↔ y_valid
# X_test  ↔ y_test


# Ev özelliklerini PyTorch tensoruna çevir.
X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_valid = torch.tensor(
    X_valid,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)



y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

y_valid = torch.tensor(
    y_valid,
    dtype=torch.float32
).reshape(-1, 1)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)


# Her özellik sütununun ortalamasını hesapla.
means = X_train.mean(
    dim=0,
    keepdim=True
)

# Her özellik sütununun standart sapmasını hesapla.
stds = X_train.std(
    dim=0,
    keepdim=True
)



X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test = (X_test - means) / stds

# X_train[0] ↔ y_train[0]  ev özellikleri ile fiyatlarını eşleştirir.

train_dataset = TensorDataset(
    X_train,
    y_train
)
first_X, first_y = train_dataset[0]

train_loader = DataLoader(
    train_dataset,
    #Her seferinde modele 32 ev gönder.
    batch_size=32,
    #Her epoch başında train verilerinin sırasını karıştırır.
    shuffle=True
) #1. batch → 32 ev + bu evlerin 32 gerçek fiyatı

#iter train_loaderi sırayla verebilen bir yapıya dönüşür. İterator birincinin arkasından başladığı için next koyman lazım.
X_batch, y_batch = next(iter(train_loader))

torch.manual_seed(42)

n_features = X_train.shape[1]

#İçine koyduğum katmanları sırayla çalıştır, Bir katmanın çıktısını bir katmanın girdisi yap.
model = nn.Sequential(

nn.Linear(n_features,50), #Bu katmana 50 nöron verdik, her nöron evin 8 özelliğine ayrı ağırlıklar uygular.

nn.ReLU(), #negatif değerleri 0 lar, pozitifleri olduğu gibi bırakır.

nn.Linear(50,40), #Önceki katmandan 50 sayı geldiği için giriş 50 olmalıdır.

nn.ReLU(),

nn.Linear(40,1)
)

# Nvidia ekran kartı ve CUDA kullanılabiliyorsa GPU'yu seç.
if torch.cuda.is_available():
    device = torch.device("cuda")

# Apple bilgisayarda MPS kullanılabiliyorsa Apple GPU'yu seç.
elif torch.backends.mps.is_available():
    device = torch.device("mps")

# GPU kullanılamıyorsa işlemciyi seç.
else:
    device = torch.device("cpu")

#CPU: Bilgisayarın genel amaçlı işlemcisidir. Küçük modellerde yeterlidir.
#GPU: Çok sayıda küçük hesabı aynı anda yapabilen işlemcidir.
#CUDA: Nvidia ekran kartlarının hesaplama sistemidir.
#MPS: Apple GPU

model.to(device)
criterion = nn.MSELoss()

learning_rate = 0.02

## SGD, modeldeki bütün ağırlıkları ve bias değerlerini gradientlere göre günceller.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

n_epochs = 20

#Mini-Batch Eğitim Fonk.

def train_model(
        model,
        optimizer,
        criterion,
        train_loader,
        n_epochs
):
    #Modeli eğitim moduna geçirir.
    model.train()

    for epoch in range(n_epochs):

        total_loss = 0.0

        #DataLoader iki tane şey return eder.
        for X_batch, y_batch in train_loader:

            #Batchleri cihaza taşı, Gpudaysa gpuya gider, cpudaysa cpuya gider.
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            #Bu ev için 32 fiyat tahmini üret.
            y_pred = model(X_batch)

            loss = criterion(
                y_pred,
                y_batch
            )

            #Modeldeki bütün ağırlıkların gradientlerini hesapla.
            loss.backward()
            #Gradientleri kullanarak bütün ağırlıkları güncelle.
            optimizer.step()

            #Bu batch'in loss değerlerini epoch toplamına ekle
            total_loss += loss.item()

            # Epoch boyunca hesaplanan batch loss'larının ortalamasını al.
        mean_loss = total_loss / len(train_loader)


        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"Loss: {mean_loss:.4f}"
        )



valid_dataset = TensorDataset(
    X_valid,
    y_valid
)
#Validationda eğitim yapılmadığı için gradient ayarı yapılmaz.
valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,

    # Validation sırasında eğitim yapılmadığı için
    # verilerin sırasını karıştırmamız gerekmez.
    shuffle=False
)

def evaluate_rmse(model, data_loader):

    # Modeli değerlendirme moduna geçir.
    model.eval()

    total_squared_error = 0.0
    total_predictions = 0

    # Değerlendirme sırasında eğitim yapmıyoruz.
    # Bu yüzden gradient hesaplamaya gerek yok.
    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            # Verileri modelle aynı cihaza taşı.
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Modelin fiyat tahminlerini üret.
            y_pred = model(X_batch)

            # Bu batch'teki kare hataların toplamını hesapla.
            squared_errors = (y_pred - y_batch) ** 2

            total_squared_error += squared_errors.sum().item()

            # Bu batch'te kaç tane fiyat tahmini bulunduğunu say.
            total_predictions += y_batch.numel()

    # Bütün validation verisinin MSE değeri.
    mse = total_squared_error / total_predictions

    # Karekök alarak RMSE elde et.
    rmse = mse ** 0.5

    return rmse
#Eğitim çağrısından sonra fonksiyonu kullan:
train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    n_epochs
)

validation_rmse = evaluate_rmse(
    model,
    valid_loader
)

print(
    f"Validation RMSE: {validation_rmse:.4f}"
)