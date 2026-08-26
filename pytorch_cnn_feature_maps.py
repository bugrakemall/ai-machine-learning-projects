from sklearn.datasets import load_sample_images
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

sample_images = np.stack(load_sample_images()["images"])

sample_images = torch.tensor(
    sample_images,
    dtype=torch.float32
) / 255  #Pixellerin aralığını 0-1 e getirir.

#Conv2d görüntüleri şu sırada bekler (batch, channel, height, width) eski üçüncü artık birinci boyut vs.
sample_images_permuted = sample_images.permute(0, 3, 1, 2)

cropped_images = T.CenterCrop((70, 120))(sample_images_permuted) #resmin merkezinden 70h 120w image aldık

torch.manual_seed(42)

conv_block = nn.Sequential(
    nn.Conv2d(
        in_channels=3, #RGB 
        out_channels=32, #32 tane filter/kernel deneyecek.
        kernel_size=7 # 7x7 şeklinde incelemeye başlayacak.
    ),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2), #pool=nn.MaxPool2d(kernel_size=2) #Feature mapi parçalar 2x2 bölgeler haline getirir ve her parçadaki en büyük değeri tutar.
    nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        padding="same"
    ),
    #[out_channels, in_channels, kernel_height, kernel_width] 64,32,3,3
    #bunun shape
)

fmaps = conv_block(cropped_images) #kırpılmış resmi modele veriyor.

#Feature map: Bir kernelin görüntünün her yerini taradıktan sonra oluşan sonuç tablosudur
#Kernel parçayı inceleyen ağırlık tablosu.
#Kernel aradığı özelliktir; feature map ise bu özelliğin görüntünün nerelerinde ve ne kadar bulunduğunu gösteren sonuçtur.
#32 kernel olduğundan bias 32 dir.


print("Girdi:", cropped_images.shape)
print("Çıktı:", fmaps.shape)
print("Ağırlık:", conv_block[0].weight.shape)
print("Bias:", conv_block[0].bias.shape)