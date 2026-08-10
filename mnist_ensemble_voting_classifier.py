from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

#RandomForestClassifier birden fazla karar ağacı kullanarak karar verir.
#DecisionTreeClassifier tek bir karar ağacı ile karar verir.
print("started")
X,y = fetch_openml(
    "mnist_784", #784 çünkü 28*28 boyutlarında
    version=1,#Direkt birinci sürümü indirsin diye.
    return_X_y=True, #True diyerek X ve y return eder.
    as_frame=False #False diyerek NumPy array olarak return olur.
)

#X numara resimleri, y answers

X = X/255.0 #Rakam görüntülerinin piksel değerlerini 0-1 arasına getiriyor.
y = y.astype(np.uint8) #Cevapları [0,9] arasında tam sayılara dönüştürür. astype türü değiştir demek cevaplar string olarak geliyor.

X_train_valid, X_test, y_train_valid,y_test = train_test_split( #70k tane verinin 10k tanesini test için ayırır.
    X, #Genel train seti
    y, #Genel cevaplar
    random_state=42,
    test_size=10_000, #elimizde 70k data var bunun sadece 10k sını teste koyuyor 60k sı train e gidiyor
    stratify=y #sınıf dağılımını mümkün olduğunca koru.
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, 
    y_train_valid,
    random_state=42,
    stratify=y_train_valid,
    test_size=10_000
)

random_forest_clf = RandomForestClassifier( #En iyi bölme sınırlarını arar
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)
extra_tree_clf = ExtraTreesClassifier( #Bölme sınırlarını daha rastgele seçer.
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)
svc_clf = SVC( #SVC, farklı rakam görüntülerini birbirinden ayıran sınırları öğrenir. Bu sınırların yerini özellikle birbirine benzeyen
    #ayırması zor görüntüler belirler; bunlara support vectors denir.
    probability=True
)
hard_voting_clf = VotingClassifier(
    estimators=[
        ("random_forest",random_forest_clf),
        ("extra_tree",extra_tree_clf),
        ("svc",svc_clf)
    ],
    voting="hard", #hard voting direkt oylamayla karar verir #soft voting ortalama alır en yüksek olana karar verir.
    n_jobs=-1
)
hard_voting_clf.fit(X_train, y_train)

hard_voting_clf.voting = "soft" #böyle extra eğitmeden hardı softa değiştirirsin.
soft_voting_predictions = hard_voting_clf.predict(X_valid) #sonra tahmin ettirirsin.


random_forest_clf.fit(X_train,y_train)
extra_tree_clf.fit(X_train,y_train)
svc_clf.fit(X_train, y_train)

rf_valid_predictions = random_forest_clf.predict(X_valid) #10k tane validation görüntüsü için tahminler üretildi.
extra_tree_predictions = extra_tree_clf.predict(X_valid)
svm_valid_predictions = svc_clf.predict(X_valid)
hard_voting_predictions = hard_voting_clf.predict(X_valid)


rf_valid_accurancy = accuracy_score(
    rf_valid_predictions, #10k tane modelin tahmini
    y_valid #10k tane gerçek cevap.
)
extra_tree_clf_valid_accurancy = accuracy_score(
    extra_tree_predictions, #10k tane modelin tahmini
    y_valid #10k tane gerçek cevap.
)   
hard_voting_accuracy = accuracy_score(
    y_valid,
    hard_voting_predictions
)
svm__accuracy = accuracy_score(
    y_valid,
    svm_valid_predictions
)
soft_voting_accuracy = accuracy_score(
    y_valid,
    soft_voting_predictions
)

print("rft validation accuracy:", rf_valid_accurancy)
print("eft validation accuracy:", extra_tree_clf_valid_accurancy)
print("hv validation accuracy:", hard_voting_accuracy)
print("SVM validation accuracy:", svm__accuracy)
print("Soft Voting validation accuracy:",soft_voting_accuracy)

best_individual_test_accuracy = max(
    rf_valid_accurancy,
    extra_tree_clf_valid_accurancy,
    svm__accuracy
)

#Hard voting ile en iyi tekil model arasındaki farkı hesaplar.
hard_voting_improvement = (
    hard_voting_accuracy
    - best_individual_test_accuracy
)
