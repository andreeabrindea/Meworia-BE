import json
import pandas as pd


def get_symptoms():
    data = pd.read_csv('/Users/andreea/PycharmProjects/Meowria/archive/Training.csv')
    data = data.drop(columns=["prognosis"])
    header_list = list(data.columns)

    list_of_symptoms = []
    for i in header_list:
        list_of_symptoms.append(i)

    # Print the list of unique symptoms
    a = list(set(list_of_symptoms))

    # Write the symptom list to a JSON file
    with open('symptoms.json', 'w') as f:
        json.dump({'symptoms': a}, f)


get_symptoms()
