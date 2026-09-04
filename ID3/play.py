#Write a program to demonstrate the working of the decision tree based on ID3 algorithm. Use an 
#appropriate data set for building the decision tree and apply this knowledge to classify a new sample. 
import csv
from sklearn.tree import DecisionTreeClassifier

with open("play.csv", "r") as file:
    data = list(csv.reader(file))

X = []
y = []

for row in data[1:]:
    X.append([0 if row[0] == "Sunny" else 1])
    y.append(0 if row[1] == "No" else 1)

model = DecisionTreeClassifier(criterion="entropy")

model.fit(X, y)

prediction = model.predict([[0]])

if prediction[0] == 1:
    print("Play Tennis: Yes")
else:
    print("Play Tennis: No")