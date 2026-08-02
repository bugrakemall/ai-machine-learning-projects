from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,ShuffleSplit,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import numpy as np
from scipy.stats import mode

coordinates,answers = make_moons(
    n_samples=10_000,
    noise=0.4, #noktaları dağıtarak sınıflandırmayı zorlaştırır.
    random_state=42
)

coordinates_train,coordinates_test,answers_train,answers_test = train_test_split(
    coordinates,
    answers,
    random_state=42,
    test_size=0.20,
    stratify=answers
)

parameters = {
    "max_leaf_nodes":range(2,101)
}

tree_clf = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator= tree_clf, #Hangi model ve ayarlarını deneyeceğini belirtir.
    param_grid= parameters, #Hangi parametreleri uygulayacağını belirtir.
    cv=3, 
    scoring="accuracy" #corret answers / all answers
)

grid_search.fit(coordinates_train, answers_train)

best_tree = grid_search.best_estimator_ #en iyi alınan sonucu burada tutuyor.

test_predictions = best_tree.predict(coordinates_test) #test verilerini tahmin ettiriyor

test_accurancy = accuracy_score(
    answers_test, #cevaplar
    test_predictions #modelin tahminleri
)
###################
n_trees = 1000
n_instances = 100

shuffle_split = ShuffleSplit(
    n_splits=n_trees, #Bunu 1000 kere tekrarla. totalde 1000 kere çalışır.
    train_size=n_instances, #Rastgele 100 tane veri seç, 8000 tane verinin içinden.
    random_state=42
)
forest = []
for tree_index in range(n_trees):
    new_tree = clone(best_tree)
    forest.append(new_tree)

tree_accuracies  = []

for tree, (selected_indices, remaining_indices) in zip(
    forest,
    shuffle_split.split(coordinates_train)
):
    mini_coordinates = coordinates_train[selected_indices]
    mini_answers = answers_train[selected_indices]

    tree.fit(
        mini_coordinates,
        mini_answers
    )

    tree_predictions = tree.predict(coordinates_test)

    tree_accuracy = accuracy_score(
        answers_test,
        tree_predictions
    )

    tree_accuracies.append(tree_accuracy)
#########
#1000 ağacın tahminlerini tek bir matriste toplayacağız.
all_predictions = np.empty(
    shape=(n_trees,len(coordinates_test)), #1k row, 2k column boyutunda matris oluşturur.
    dtype=np.uint8 #0 veya 1 sayılarını kullanır. uint8 0-255 arasını saklar.
)

for tree_index, tree in enumerate(forest): #enumerate satırları sıralandırır başına rakam koyar.
    all_predictions[tree_index] = tree.predict(coordinates_test) #burada modelin tahminlerini empty arraye kaydediyor.

majority_predictions = mode(
    all_predictions,  # 1000 ağacın 2000 test noktası için tahminleri

    axis=0,  # 0 kullanınca sütun sütun 1 kullanınca satır satır

    keepdims=False  # Eritilen satır eksenini tamamen kaldırır

).mode  # mode() sonucunun içinden çoğunluk değerlerini alır

forest_accuracy = accuracy_score(
    answers_test,          # 2000 gerçek cevap
    majority_predictions   # Ormanın 2000 çoğunluk tahmini
)

print("Tek büyük ağacın doğruluğu:", test_accurancy)

print(
    "100 örnekle eğitilen küçük ağaçların ortalaması:",
    np.mean(tree_accuracies)
)

print("Ormanın doğruluğu:", forest_accuracy)