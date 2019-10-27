import pandas as pd
import os

df = pd.read_csv(os.path.join('data', 'heart.csv'))

print(df.head())