import json

import pandas as pd


def get_symptoms():
    # Load the dataset
    data = pd.read_csv('/Users/andreea/PycharmProjects/Meowria/archive/dataset.csv')
    data = data.drop(columns=["Disease"])

    # Filter out rows with NaN values and replace them with 'unknown'
    data = data.fillna("unknown")

    symptoms = data.values

    # Convert the array to a list of symptoms
    list_of_symptoms = list(symptoms.flatten())

    k = []
    for i in list_of_symptoms:
        j = i.replace(' ', '')
        #j = j.replace('_', " ")
        k.append(j)

    # Print the list of unique symptoms
    a = list(set(k))
    a.remove('unknown')

    # Write the symptom list to a JSON file
    with open('symptoms.json', 'w') as f:
        json.dump({'symptoms': a}, f)
