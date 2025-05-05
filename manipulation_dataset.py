#%%

import pandas as pd
from utilities import preprocess
import ast
DATASET = 'dataset_'
DATASET += '28_04'
df = pd.read_csv(DATASET + '.csv')

user_id_counts = df['user_id'].value_counts()

NEW_COLUMN = 'user_id_count'

df[NEW_COLUMN] = df['user_id'].map(user_id_counts)
# df.to_csv(NAME + '_with_'+NEW_COLUMN +'.csv', index=0)

df['text_preprocessed'] = df['text'].apply(preprocess)

# Crear la nueva columna con la longitud de caracteres después del preprocesado
df['text_preprocessed_length'] = df['text'].apply(preprocess).apply(len)
df['text_original_length'] = df['text'].apply(len)
df['text_length_ratio'] = df['text_preprocessed_length'] / df['text_original_length']

# Creation of hashtags as nodes
#%%
def normalize_column(value):
    # if pd.isna(value):
    #     return []
    if isinstance(value, list):
        return [str(x).upper().strip() for x in value if x.strip()]
    elif isinstance(value, str):
        return [value.upper().strip()]
    else:
        return []

def string_to_list(string):
    return ast.literal_eval(string)

#%%
df['hashtags'] = df['hashtags'].apply(string_to_list)
df['hashtags'] = df['hashtags'].apply(normalize_column)
df = df[df['hashtags'].map(len) > 0]

df_exploded = df.explode('hashtags')
top_n = 50  # Número de hashtags más comunes
top_hashtags = df_exploded['hashtags'].value_counts().head(top_n).index
df_filtered = df_exploded[df_exploded['hashtags'].isin(top_hashtags)]

# df_majority = (
#     df_filtered.groupby('hashtags')['sentiment']
#     .agg(lambda x: pd.Series.mode(x)[0])  # modo (sentimiento más frecuente)
#     .reset_index()
#     .rename(columns={'hashtags': 'hashtag', 'sentiment': 'majority_sentiment'})
# )

df_majority = (
    df_filtered.groupby('hashtags')
    .agg(
        majority_sentiment=('sentiment', lambda x: pd.Series.mode(x)[0]),
        user_ids=('user_id', lambda x: list(x.unique()))
    )
    .reset_index()
    .rename(columns={'hashtags': 'hashtag'})
)
#%%
edges_df = df_filtered[['user_id', 'hashtags']].dropna().drop_duplicates()
edges_df.columns = ['user', 'hashtag']  

selection_test = edges_df.sample(10000)

import networkx as nx
import matplotlib.pyplot as plt

# Crear un grafo bipartito
B = nx.Graph()

# Añadir nodos por tipo
edges_df = selection_test.copy()
users = edges_df['user'].unique()
hashtags = edges_df['hashtag'].unique()

B.add_nodes_from(users, bipartite='users', color = 'blue')
B.add_nodes_from(hashtags, bipartite='hashtags', color = 'red')

# Añadir aristas: cada usuario conectado al hashtag que usó
B.add_edges_from(edges_df.itertuples(index=False) )

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(B, k=0.15, iterations=20)

nx.draw(B, pos, node_size=15, with_labels=False)
plt.title("Red bipartita: usuarios y hashtags")
plt.show()

#%%
DATE = '28_04'
df.to_csv('dataset_'+DATE+'.csv', index=0)

# %%
