import torch
import torch.nn as nn
from torch.nn import functional as F

x = torch.tensor([
    [1.0, 0.0],  # A tokenı
    [0.0, 1.0],  # B tokenı
    [1.0, 1.0]   # C tokenı
])

torch.manual_seed(1337)

class SmallHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.key = nn.Linear(2, 2, bias=False)
        self.query = nn.Linear(2, 2, bias=False)
        self.value = nn.Linear(2, 2, bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(3, 3)))
        
        #Alt üçgen ve köşegen kısmını 1 yapar. Geri kalan yerler 0 olur.

    def forward(self, x): #tensorleri buna vereceğiz.

        k = self.key(x) #Bende ne var
        q = self.query(x) #Ne arıyorum
        v = self.value(x) #Sana aktaracağım bilgi.

        scores = q @ k.transpose(-2,-1)
        scores = scores *(k.shape[-1]**-0.5) #Query-key çarpımından sonra softmax yapmadan önce aşırı keskinleşmesin diye
        #key uzunluğunun kareköküne bölünür. 
        T= x.shape[0]

        izin_tablosu = self.tril #alt üçgenli tablo
        yasak_yerler = izin_tablosu == 0 #PyTorcha özel 0 ları bulucu.
        masked_scores = scores.masked_fill(yasak_yerler,float("-inf")) #Sıfırları sonsuz yapıyozki softmaxta olasılığı 0 olsun.
        #Bunları sıfırları -inf yapmak içni yaptık.
        
        attention_weights = F.softmax(masked_scores,dim=-1)

        out = attention_weights @ v
         
        return k, q, v,scores,masked_scores,attention_weights,out

head = SmallHead()

k, q, v, scores,masked_scores,attention_weights,out = head(x)

#print(head.tril)
print(masked_scores)
print(attention_weights)
print(out)

