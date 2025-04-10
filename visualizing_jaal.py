import pandas as pd
from pyvis.network import Network

DATASET = 'tweets_with_groups_and_urls_all_without_RT'
CSV = '.csv'
data = pd.read_csv(DATASET+CSV)

from jaal.datasets import load_got
#load the data
edge_df, node_df = load_got()
#init Jaal and run server
Jaal(edge_df, node_df).plot()