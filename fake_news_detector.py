import re
import string
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["class"] = 0
true["class"] = 1

fake_manual = fake.tail(10).copy() #save a few articles for manual testing later
true_manual = true.tail(10).copy()

fake = fake.iloc[:-10]
true = true.iloc[:-10]

data = pd.concat([fake, true], ignore_index=True)
data = data.drop(columns=["title", "subject", "date"])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\W", " ", text)
    return text


data["text"] = data["text"].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["class"],
    test_size=0.25,
    random_state=42,
)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\n{name}")
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")
    print(classification_report(y_test, predictions))


def output_label(value):
    return "Not A Fake News" if value else "Fake News"


def manual_testing(news):

    cleaned = clean_text(news)
    transformed = vectorizer.transform([cleaned])

    print()

    for name, model in models.items():
        prediction = model.predict(transformed)[0]
        print(f"{name}: {output_label(prediction)}")


while True:
    article = input("\nEnter news article (or type 'quit'): ")

    if article.lower() == "quit":
        break

    manual_testing(article)