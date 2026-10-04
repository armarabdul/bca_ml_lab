import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("data.csv")

X = data[["Age", "Salary"]]

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)

data["Cluster"] = kmeans.fit_predict(X)

print(data)
