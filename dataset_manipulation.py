import json

import pandas as pd

df = pd.read_csv('/Users/andreea/PycharmProjects/Meowria/archive/dataset.csv')

with open('symptoms.json') as f:
    content = json.load(f)
symptoms = content['symptoms']

disease_df = pd.DataFrame(columns=['Disease'] + symptoms)

for index, row in df.iterrows():
    disease = row['Disease']
    symptom_values = [1 if symptom in row.values[1:] else 0 for symptom in symptoms]
    disease_df.loc[index] = [disease] + symptom_values

disease_df.to_csv('new_dataset.csv', index=False)
