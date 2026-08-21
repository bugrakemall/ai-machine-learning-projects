import torch
from torch import nn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()

X = housing.data
y = housing.target

#Verinin %20 sini test için ayır son ikisini yani. %80 i train
#Validation sayesinde modelin overfitting yapıp yapmadığını da anlayabiliriz.
X_train_valid,y_train_valid,X_test,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

#yüzde 80 lik verinin yüzde 20 si son ikisine gidiyor
X_train,X_valid,y_test,y_valid = train_test_split(
    X_train_valid,
    y_train_valid,
    test_size=0.20,
    random_state=42
)

X_train = torch.tensor(X_train,dtype=torch.float32) #veriyi PyTorch formatına çevirir. sayıların tipini 32-bit ondalıklı sayı yapar.
X_valid = torch.tensor(X_valid,dtype=torch.float32)#PyTorch modelleri giriş verisini Tensor olarak bekler.
X_test = torch.tensor(X_test,dtype=torch.float32)


# Önce: (örnek_sayısı,)
# Sonra: (örnek_sayısı, 1)
y_train_valid = torch.tensor(
    y_train_valid,
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

means = X_train.mean(dim=0,keepdim=True)
std = X_train.std(dim=0,keepdim=True)

X_train = (X_train-means) / std
X_valid = (X_valid-means) / std
X_test = (X_test-means) / std

torch.manual_seed(42)

# (örnek_sayisi,özellik_sayisi)'ndan özellik sayisi 8 i alır.
n_features = X_train.shape[1]


#Otomatik olarak her özellik için bir ağırlık ve bir adet bias oluşturur.
model = nn.Linear(
    in_features=n_features, #Her evde 8 özellik vardır.
    out_features=1 #Her ev için tek fiyat tahmini üretilecek
)

criterion = nn.MSELoss()

learning_rate = 0.4
n_epochs = 20

# SGD, modelin weight ve bias değerlerini gradientlere bakarak günceller.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

for epoch in range(n_epochs):

    #Önceki epochtan kalan gradları temizle.
    optimizer.zero_grad()

    #Arka planda şunu yapar X_train @ model.weight.T + model.bias
    y_pred = model(X_train)

    #Tahminle gerçek fiyat arasındaki MSE'yi hesapla.
    loss = criterion(
        y_pred,
        y_test
    )

    #w ve b' değerlerinin gradientleri hesapla.
    loss.backward()
    optimizer.step() #weight ve biasları güncelle.  w -= learning_rate * w.grad

    print(
        f"Epoch {epoch + 1}/{n_epochs}, "
        f"Loss: {loss.item():.4f}"
    )

#Validation setini değerlendir.
#Burada eğitim yapmadığımız için gradient hesaplamayı kapatıyoruz.
with torch.no_grad():
    #Validation veri setini değerlendiriyoruz.
    #Tahmin. Model tahmin ettiriyor. Formülü kullanıyoruz lineer olan.
    y_valid_pred = model(X_valid)

    valid_mse = criterion(
        y_valid_pred,
        y_valid
    )

    valid_mse = torch.sqrt(valid_mse) #rmse
    print("\nValidation RMSE:",valid_mse.item())


#Test setini değerlendir
with torch.no_grad():
    y_test_pred = model(X_test)

    test_mse = criterion(
        y_test_pred,
        y_test
    )

    # Test RMSE değerini hesapla.
    test_rmse = torch.sqrt(test_mse)



# 11. İLK 3 TEST EVİ İÇİN TAHMİN YAP

X_new = X_test[:3]

with torch.no_grad():
    new_predictions = model(X_new)


print("\nİlk 3 tahmin:")
print(new_predictions)

print("\nİlk 3 gerçek cevap:")
print(y_test[:3])


# model.weight'ın şekli (1, 8)'dir.
#
# .T kullanarak önceki kodumuzdaki gibi
# (8, 1) biçiminde gösteriyoruz.
print("\nModelin öğrendiği ağırlıklar:")
print(model.weight.T)

print("\nModelin öğrendiği bias:")
print(model.bias)