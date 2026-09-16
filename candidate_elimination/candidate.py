import pandas as pd

data = pd.read_csv("data.csv")

S = ['0', '0', '0', '0']
G = ['?', '?', '?', '?']

for i in range(len(data)):
    x = list(data.iloc[i, :-1])
    y = data.iloc[i, -1]

    if y == "Yes":
        for j in range(4):
            if S[j] == '0':
                S[j] = x[j]
            elif S[j] != x[j]:
                S[j] = '?'

print("Specific Boundary:", S)
print("General Boundary:", G)