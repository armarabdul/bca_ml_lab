import pandas as pd
from sklearn.mixture import GaussianMixture

data = pd.read_csv("data.csv")

model = GaussianMixture(n_components=2)

model.fit(data)

data["Cluster"] = model.predict(data)

print(data)