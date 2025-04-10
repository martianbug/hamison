from is_rt import is_rt
import pandas as pd

DATASET = 'tweets_with_groups_and_urls_all'
CSV = '.csv'

df = pd.read_csv(DATASET+CSV)

df_clean = df[~df.apply(is_rt, axis=1)].reset_index(drop=True)


WR = '_without_RT'
df_clean.to_csv(DATASET + WR + CSV, index =0)
