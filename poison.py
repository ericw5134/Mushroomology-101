import pandas as pd

data = pd.read_csv('secondary_data.csv', sep=';')


poisonous = data[data['class'] == 'p']

common_values = poisonous.mode().iloc[0]

for column in poisonous.columns:
    print(f"Most common value for {column}: {common_values[column]}")
