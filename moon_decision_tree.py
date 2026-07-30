from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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
print(answers_test[:10])
print(test_predictions[:10])
print(test_accurancy)