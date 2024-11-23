import pandas as pd
import json
import numpy as np
with open('mapping.json', 'r') as json_file:
    data_dict = json.load(json_file)

file = 'test1.csv'
target = "Race"
output_file = 'test1_onehot.csv'
df = pd.read_csv(file)
for key in data_dict.keys():
    print(key)
    if (key==target):
        continue
    possible_values = data_dict[key]
    one_hot = pd.get_dummies(df[key])
    one_hot = one_hot.reindex(columns=sorted(possible_values),fill_value=False)
    one_hot = one_hot.iloc[:,1:]
    one_hot = one_hot.astype(int)
    one_hot.columns = [f"{key}_{col}" for col in one_hot.columns]
    one_hot_array = one_hot.values
    column_index = df.columns.get_loc(key)
    df = df.drop(columns=[key])
    df_array = df.values
    columns_before = df_array[:, :column_index]
    columns_after = df_array[:, column_index:]
    new_columns = list(df.columns[:column_index]) + list(one_hot.columns) + list(df.columns[column_index:])
    combined_array = np.hstack([df_array[:, :column_index], one_hot_array, df_array[:, column_index:]])
    df = pd.DataFrame(combined_array, columns=new_columns)
    print(df.shape)

df.to_csv(output_file,index=False)


