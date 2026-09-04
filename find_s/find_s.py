#Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a
#given set of training data samples. Read the training data from a .CSV file.

import csv

with open("data.csv", "r") as file:
    data = list(csv.reader(file))

header = data[0]
examples = data[1:]

hypothesis = ["?"] * (len(header) - 1)

for row in examples:
    if row[-1] == "Yes":
        for i in range(len(hypothesis)):
            if hypothesis[i] == "0":
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = "?"

print("Most Specific Hypothesis:")
print(hypothesis)