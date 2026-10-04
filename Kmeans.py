import pandas as pd
from sklearn.cluster import KMeans

# Read CSV file
data = pd.read_csv("data.csv")

# Select numerical columns
X = data.select_dtypes(include=['number'])

# Create K-Means model
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)

# Perform clustering
data["Cluster"] = kmeans.fit_predict(X)

# Display result
print(data)
