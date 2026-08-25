from torchvision.transforms import v2
import torch
import torchvision
from torch.utils.data import DataLoader,random_split
from torch import nn

#Görüntüyü modele vermeden önce hazırlamak için kullanılan araçları içerir
#Compose Sequential gibi işlemleri sıraya koyar.
############################################################################
transform = v2.Compose([

v2.ToImage(), #image'nin tensoru
v2.ToDtype(
    torch.float32, #float32 olsun
    scale=True  #pixel değerlerini 0-1 arasına getirir.
)
])
############################################################################
#Fashion-MNIST veri setini hazırla.
full_train_data = torchvision.datasets.FashionMNIST(
root="datasets", #bilgisayarda veri setinin saklanacağı klasör
train=True,
download=True, #dataset yoksa internetten indirir.
transform=transform #üstteki işlemleri kullan
)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets",
    train=False,
    download=True,
    transform=transform
)
############################################################################
generator = torch.Generator().manual_seed(42) #PyTorch random sayı generator.

#train_test_split NumPy arraylar için kullanılır. Bu PyTorch array
train_data, valid_data = random_split(
full_train_data,
#55k train, 5k valid datas
[55_000,5_000],
generator=generator
)

train_loader = DataLoader(
    train_data,
    batch_size=32,  #Her batch'ın içine 32 tane resim koy.
    shuffle=True #Datayı karıştırır
)
valid_loader = DataLoader(
    valid_data,
    batch_size=32,
    shuffle=False
)
test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

X_batch,y_batch = next(iter(train_loader)) #DataLoader iki tane değer return eder.
#print(X_batch) #32 kıyafet resimlerinin pixelleri

class ImageClassifier(nn.Module):

    def __init__(self):
        super().__init__() #nn Moduleden miras alıyoz ve kurulumunun yapılması gerekiyor.

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,300),
            nn.ReLU(),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.Linear(100,10) #10 tane sınıf olduğu için.
        )
    def forward(self,X):
        scores = self.mlp(X) # Görüntüyü yukarıdaki katmanlardan sırayla geçir.
        return scores

torch.manual_seed(42)

model = ImageClassifier()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device) #Modelin bütün ağırlıklarını seçtiğimiz işlem cihazına taşır. Model ve veri aynı cihazda olmak zorundadır.

criterion = nn.CrossEntropyLoss() #10 farklı seçenek arasından puan veriyor en çok puan alan cevap oluyor.

optimizer = torch.optim.SGD( #Modelin öğrendiği ağırlıkları güncelleyecek optimizeri oluşturur. Ağırlıkları değiştirir.
    model.parameters(), #Model içinde ağırlıkları ve bias değerini bulur.
    lr= 0.1 #yeni ağırlık = eski ağırlık - lr*gradient
)

n_epochs = 10

def evaluate_accuracy(model,data_loader):
    model.eval() #modeli değerlendirme moduna aldık.

    correct,total = 0,0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch=X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch) #32 resmi modele verir. Model her resim için 10 sınıf skoru üretir.
            predictions = logits.argmax(dim=1)  #satırdaki en büyük skorun indexini bulur.

            correct += (predictions == y_batch).sum().item() #Doğruları sayıyor.
            total += y_batch.size(0) #toplam batchtaki kıyafet sayısını yazıyor
    return correct / total

for epoch in range(n_epochs): #eğitim döngüsü
    model.train()

    total_loss = 0.0
    total_images = 0

    for X_batch, y_batch in train_loader: #train_loader = train set
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad() #öncekilerden kalan gradleri temizler

        logits = model(X_batch) #Modele değerleri vermeye başlar.
        loss = criterion(logits,y_batch) #Modelin ürettiği skorlar ile gerçek cevapları karşılaştırıp hata değeri hesaplar.

        loss.backward() #Geriye doğru giderek bu ağırlığın bu hataya etkisi ne oldu diye sorar.
        optimizer.step() #Bir önceki satırda hesaplanan gradientleri kullanarak ağırlıkları günceller.

        total_loss += loss.item() * X_batch.size(0) #loss.item hataların ortalamasıdır.
        total_images += X_batch.size(0)

    mean_loss = total_loss / total_images
    valid_accuracy = evaluate_accuracy(model,valid_loader)
    print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"Loss: {mean_loss:.4f} | "
            f"Validation accuracy: {valid_accuracy:.4f}"
        )


class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

model.eval()

X_test_batch, y_test_batch = next(iter(test_loader))

X_test_batch.to(device)
y_test_batch.to(device)

with torch.no_grad():
    logits = model(X_test_batch)
    predictions = logits.argmax(dim=1) #dim=1 dediğimiz için her satırda soldan sağa bakar ve en büyük skorun sütun numarasını döndürür.

for index in range(5):
    predicted_number = predictions[index].item()
    actual_number = y_test_batch[index].item()