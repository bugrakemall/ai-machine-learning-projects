import torch

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

w= torch.rand(
    (n_features,1), #8 tane rastgele değer ve bir özellik olsun.
    requires_grad=True 
)
b = torch.tensor(
    0.0, #başlangıç değeri
    requires_grad=True #Eğitim sırasında bu değerin hataya etkisini hesapla.
)

learning_rate = 0.4 #Modelin w ve b değerlerini her güncellemede ne kadar değişeceğini belirler.
n_epochs = 20 #Modelin bütün X_train verisini 20 kez kullanarak eğitim yapacağını söyler.

for epoch in range(n_epochs):
    #ev özellikleri yatay matris, ağırlıkları dikey matris çarpınca tahmin ortaya çıkıyor.
    y_pred = X_train @ w + b

    #Loss func. regresyon sayı tahminlerinde kullanılır. Tahmin-gerçek cevabın karesinin ortalaması
    loss = ((y_pred-y_valid)**2).mean()

  
#  PyTorch bu tahminin w ve b kullanılarak üretildiğini biliyor. Çünkü tensorladık:
    loss.backward()

    with torch.no_grad():
        #Gradient hatanın arttığı yönü gösterir, bizde tersine giderek hatayı azaltacak şekilde gideriz.
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        #her epoch sonunda gradientleri sıfırlarız.
        w.grad.zero_()
        b.grad.zero_()

        print(
                f"Epoch {epoch + 1}/{n_epochs}, "
                f"Loss: {loss.item():.4f}" #.item() → Tensordaki tek değeri Python sayısına çıkarır
                #:.4f    → Ondalık kısmı 4 basamak gösterir
            )

with torch.no_grad():

    y_valid_pred = X_valid @ w + b

    valid_loss = (
        (y_valid_pred - y_valid) ** 2
    ).mean()

    valid_rmse = torch.sqrt(valid_loss)


print("\nValidation RMSE:", valid_rmse.item())

with torch.no_grad():

    y_test_pred = X_test @ w + b

    test_loss = (
        (y_test_pred - y_test) ** 2
    ).mean()

    test_rmse = torch.sqrt(test_loss)


print("Test RMSE:", test_rmse.item())

#ilk 3 test

X_new = X_test[:3]

with torch.no_grad():
    new_predictions = X_new @ w + b


print("\nİlk 3 tahmin:")
print(new_predictions)


print("\nİlk 3 gerçek cevap:")
print(y_test[:3])