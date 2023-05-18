import csv


def get_disease_description(disease_name):
    with open('symptom_Description.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Disease'] == disease_name:
                return row['Description']
    return 'Description not found'


def get_disease_precaution(disease_name):
    with open('symptom_precaution.csv', 'r') as file:
        reader = csv.DictReader(file)
        precautions = []
        for row in reader:
            if row['Disease'] == disease_name:
                precautions.append(row['Precaution_1'])
                precautions.append(", ")
                precautions.append(row['Precaution_2'])
                precautions.append(", ")
                precautions.append(row['Precaution_3'])
                precautions.append(", ")
                precautions.append(row['Precaution_4'])
    if len(precautions) > 0:
        return precautions

    return 'Precautions not found'
