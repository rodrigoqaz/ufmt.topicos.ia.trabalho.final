import pandas as pd

df = pd.read_csv("datasets/test_data.csv")
df = df[["title", "abstract"]]

df.to_json("datasets/test_data_2.json", orient="records", lines=True, force_ascii=False)