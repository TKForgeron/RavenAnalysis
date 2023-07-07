import os

import pandas as pd

output_file = os.getcwd() + "/data/processed/all_observations.csv"
path = os.getcwd() + "/data/raw"
files = os.listdir(path)

files_xls = [f"{path}/{f}" for f in files if f[-4:] == "xlsx"]
# print(files_xls)

df = pd.DataFrame()

for f in files_xls:
    data = pd.read_excel(f, 0, engine="openpyxl")
    df = pd.concat([df, data], axis=0)

df.to_csv(output_file, index=False)
