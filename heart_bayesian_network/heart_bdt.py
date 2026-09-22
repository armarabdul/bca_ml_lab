import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

# 1. Load dataset
data = pd.read_csv("heart.csv")

# 2. Select required columns
data = data[["age", "cp", "thalach", "target"]]

# 3. Convert numerical values into categories
data["age"] = pd.cut(
    data["age"],
    bins=[0, 40, 60, 100],
    labels=["Young", "Middle", "Old"]
)

data["thalach"] = pd.cut(
    data["thalach"],
    bins=[0, 120, 160, 250],
    labels=["Low", "Normal", "High"]
)

# 4. Create Bayesian Network
model = DiscreteBayesianNetwork([
    ("age", "target"),
    ("cp", "target"),
    ("thalach", "target")
])

# 5. Learn probabilities (default is Maximum Likelihood Estimation)
model.fit(data)

# 6. Create inference engine
inference = VariableElimination(model)

# 7. Give patient information
result = inference.query(
    variables=["target"],
    evidence={
        "age": "Middle",
        "cp": 2,
        "thalach": "Normal"
    }
)

# 8. Display result
print(result)

print("\nDiagnosis:")
if result.values[1] > result.values[0]:
    print("Heart Disease")
else:
    print("No Heart Disease")