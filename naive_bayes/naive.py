import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")

le = LabelEncoder()
X = le.fit_transform(data["Weather"]).reshape(-1, 1)
y = le.fit_transform(data["Play"])

model = GaussianNB()
model.fit(X, y)

pred = model.predict(X)

print("Accuracy:", accuracy_score(y, pred) * 100, "%")