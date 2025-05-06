#%% INTRO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
DATE = '06_05'
NAME = 'dataset_' + DATE
df = pd.read_csv(NAME+'.csv')
import networkx as nx
import matplotlib.pyplot as plt

def normalize_column(value):
    if isinstance(value, list):
        return [str(x).upper().strip() for x in value if x.strip()]
    elif isinstance(value, str):
        return [value.upper().strip()]
    else:
        return []

def string_to_list(string):
    return ast.literal_eval(string)
#%% Hashtags df with COLUMN_TO_ANALYZE
df['hashtags'] = df['hashtags'].apply(string_to_list)
df['hashtags'] = df['hashtags'].apply(normalize_column)
df = df[df['hashtags'].map(len) > 0]

df_exploded = df.explode('hashtags')
top_n = 50  # Número de hashtags más comunes
top_hashtags = df_exploded['hashtags'].value_counts().head(top_n).index
df_filtered = df_exploded[df_exploded['hashtags'].isin(top_hashtags)]

df_majority = (
    df_filtered.groupby('hashtags')
    .agg(
        majority_sentiment=(COLUMN_TO_ANALYZE, lambda x: pd.Series.mode(x)[0]),
        user_ids=('user_id', lambda x: list(x.unique()))
    )
    .reset_index()
    .rename(columns={'hashtags': 'hashtag'})
)
#

edges_df = df_filtered[['user_id', 'hashtags']].dropna().drop_duplicates()
edges_df.columns = ['user', 'hashtag']  
selection_test = edges_df.sample(10000)

B = nx.Graph() # Crear un grafo bipartito

edges_df = selection_test.copy()
users = edges_df['user'].unique()
hashtags = edges_df['hashtag'].unique()

B.add_nodes_from(users, bipartite='users', color = 'blue')
B.add_nodes_from(hashtags, bipartite='hashtags', color = 'green')

# Añadir aristas: cada usuario conectado al hashtag que usó
B.add_edges_from(edges_df.itertuples(index=False))
node_colors = [B.nodes[n]['color'] for n in B.nodes()]

plt.figure(figsize=(20, 14))
pos = nx.spring_layout(B, k=0.15, iterations=20)

hashtag_nodes = [n for n in B.nodes if B.nodes[n].get('bipartite') == 'hashtags']
user_nodes = [n for n in B.nodes if B.nodes[n].get('bipartite') == 'users']
# Dibujar usuarios (azul)
nx.draw_networkx_nodes(B, pos,
                       nodelist=user_nodes,
                       node_color='blue',
                       node_size=10,
                       alpha=0.6)

# Dibujar hashtags (verde)
nx.draw_networkx_nodes(B, pos,
                       nodelist=hashtag_nodes,
                       node_color='green',
                       node_size=50,
                       alpha=1)

nx.draw_networkx_edges(B, pos,
                       width=0.3,
                       alpha=0.2,
                       edge_color='gray')

labels = {n: n for n in hashtag_nodes}
nx.draw_networkx_labels(B, pos, labels,
                        font_size=5,
                        font_color='black')

nx.draw(
    B, pos,
    node_size=30,
    node_color=node_colors,
    with_labels=False,
    edge_color='gray',
    width=0.3,         # <- más fino
    alpha=0.3          # <- más transparente
)

plt.title("Red bipartita: usuarios y hashtags")
plt.show()

#%%
# Calcular grado de cada usuario (solo nodos de tipo 'user')
user_degrees = [(n, d) for n, d in B.degree() if B.nodes[n].get('bipartite') == 'users']
top_users = sorted(user_degrees, key=lambda x: x[1], reverse=True)[:10]

top_users_df = pd.DataFrame(top_users, columns=['user_id', 'num_hashtags'])
# Añadir columna 'user_created_at' desde el df original
top_users_df = top_users_df.merge(df[['user_id', 'user_created_at']].drop_duplicates(), on='user_id', how='left')

print(top_users_df)
