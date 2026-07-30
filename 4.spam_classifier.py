from pathlib import Path
import urllib.request
import tarfile
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression


root = "https://spamassassin.apache.org/old/publiccorpus/"

DATASETS = [
    "20030228_easy_ham.tar.bz2",
    "20030228_spam.tar.bz2",
]

spam_path = Path("datasets/spam")
spam_path.mkdir(parents=True,exist_ok=True)

for filename in DATASETS:
    url = root + filename
    filepath = spam_path/filename
    if not filepath.is_file():
        urllib.request.urlretrieve(url,filepath)
    with tarfile.open(filepath) as tar:
        tar.extractall(spam_path)

list((spam_path / "easy_ham").iterdir())
list((spam_path/"spam").iterdir())

def load_emails(folder_path):
    emails = []

    for file_path in folder_path.iterdir():
        if file_path.is_file():
            email_text = file_path.read_text(encoding="latin-1")
            emails.append(email_text)
    return emails

ham_emails = load_emails(spam_path/"spam")
spam_emails = load_emails(spam_path/"easy_ham")

#print(len(ham_emails))
#print(len(spam_emails))
#print(ham_emails[0][:500])

x=ham_emails + spam_emails
y= [0]*len(spam_emails) + [1] *len(ham_emails)

X_train,X_test,Y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
spam_clf = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000, class_weight="balanced")
)

spam_clf.fit(X_train, Y_train)

y_pred = spam_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))