#%% INTRO
import pandas as pd
from utilities import string_to_list, normalize_column
import networkx as nx
import matplotlib.pyplot as plt

DATE = '06_05'
NAME = 'dataset_' + DATE
COLUMN_TO_ANALYZE = 'pysentimiento'
NUMBER_OF_NODES = 750

df = pd.read_csv(NAME+'.csv')
df['hashtags'] = df['hashtags'].apply(string_to_list)
df['hashtags'] = df['hashtags'].apply(normalize_column)
df_exploded = df.explode('hashtags')

top_n = 50  # Número de hashtags más comunes
top_hashtags = df_exploded['hashtags'].value_counts().head(top_n).index
df_filtered = df_exploded[df_exploded['hashtags'].isin(top_hashtags)]

edges_df = df_filtered[['user_id', 'hashtags']].dropna().drop_duplicates()
edges_df.columns = ['user', 'hashtag'] 

B = nx.Graph() # Crear un grafo bipartito

selection_test = edges_df.sample(NUMBER_OF_NODES)
edges_df = selection_test.copy()

users = edges_df['user'].unique()
hashtags = edges_df['hashtag'].unique()

B.add_nodes_from(users, bipartite='users', color = 'blue')
B.add_nodes_from(hashtags, bipartite='hashtags', color = 'green')

# Añadir aristas: cada usuario conectado al hashtag que usó
B.add_edges_from(edges_df.itertuples(index=False))

hashtag_nodes = [n for n in B.nodes if B.nodes[n].get('bipartite') == 'hashtags']
user_nodes = [n for n in B.nodes if B.nodes[n].get('bipartite') == 'users']
#%% Hashtags df with COLUMN_TO_ANALYZE
df = df[df['hashtags'].map(len) > 0]

df_majority = (
    df_filtered.groupby('hashtags')
    .agg(
        majority_sentiment = (COLUMN_TO_ANALYZE, lambda x: pd.Series.mode(x)[0]),
        user_ids=('user_id', lambda x: list(x.unique()))
    )
    .reset_index()
    .rename(columns={'hashtags': 'hashtag'})
)
hashtag_sentiment = (
    df_filtered.groupby('hashtags')['pysentimiento']
    .agg(lambda x: pd.Series.mode(x)[0])
    .reset_index()
    .rename(columns={'hashtags': 'hashtag', 'pysentimiento': 'majority_sentiment'})
)

# Frecuencia de cada hashtag (número de veces que aparece)
hashtag_counts = df_filtered['hashtags'].value_counts().reset_index()
hashtag_counts.columns = ['hashtag', 'count']

# Unir en un solo DataFrame
hashtag_info = pd.merge(hashtag_sentiment, hashtag_counts, on='hashtag')

#%% COLOREAR SEGÚN FECHA DE CREACIÓN DEL USUARIO Y SENTIMIENTO DEL HASHTAG
import matplotlib.cm as cm
import matplotlib.colors as colors

# Convertir 'user_created_at' a datetime si no lo es
df['user_created_at'] = pd.to_datetime(df['user_created_at'])

# Crear un diccionario de user_id → fecha (timestamp numérico)
user_dates = df[['user_id', 'user_created_at']].drop_duplicates()
user_dates['timestamp'] = user_dates['user_created_at'].astype('int64')  # nanosegundos
user_color_map = dict(zip(user_dates['user_id'], user_dates['timestamp']))

# Normalizar a escala de colores
timestamps = list(user_color_map.values())
norm = colors.Normalize(vmin=min(timestamps), vmax=max(timestamps))
cmap = cm.plasma  # puedes cambiar: 'viridis', 'plasma', etc.

# Asignar colores a los nodos de usuario
user_node_colors = [cmap(norm(user_color_map.get(n, min(timestamps)))) for n in user_nodes]

# Mapeo de sentimientos a colores
sentiment_color_map = {
    'POS': 'lightgreen',
    'NEG': 'salmon',
    'NEU': 'lightblue'
}

# Diccionario: hashtag → color
hashtag_color_dict = {
    row['hashtag']: sentiment_color_map.get(row['majority_sentiment'], 'gray')
    for _, row in hashtag_sentiment.iterrows()
}

# Diccionario: hashtag → tamaño del nodo
max_count = hashtag_info['count'].max()
hashtag_size_dict = {
    row['hashtag']: 300 * (row['count'] / max_count) + 100  # escala entre 100 y 400
    for _, row in hashtag_info.iterrows()
}
#%% DIBUJAR usando nx
plt.figure(figsize=(20, 14))
pos = nx.spring_layout(B, k=0.7, iterations=50, seed=42)

# pos = nx.circular_layout(B)
# Colores y tamaños para hashtags
hashtag_colors = [hashtag_color_dict.get(n, 'gray') for n in hashtag_nodes]
hashtag_sizes = [hashtag_size_dict.get(n, 100) for n in hashtag_nodes]

# Dibujar usuarios coloreados por antigüedad
nx.draw_networkx_nodes(B, pos,
                       nodelist=user_nodes,
                       node_color=user_node_colors,
                       node_size=40,
                       alpha=0.7)

# Dibujar hashtags (coloreados y escalados)
nx.draw_networkx_nodes(B, pos,
                       nodelist=hashtag_nodes,
                       node_color=hashtag_colors,
                       node_size=hashtag_sizes,
                       alpha=0.8)

# Dibujar edges
nx.draw_networkx_edges(B, pos,
                       width=0.3,
                       alpha=0.5,
                       edge_color='gray')

# Dibujar etiquetas de hashtags con desplazamiento
label_pos = {n: (x, y + 0.03) for n, (x, y) in pos.items() if n in hashtag_nodes}
labels = {n: n for n in hashtag_nodes}
nx.draw_networkx_labels(B, label_pos, labels,
                        font_size=10,
                        font_color='black')

plt.title("Red bipartita: usuarios coloreados por antigüedad, hashtags coloreados según sentimiento y tamaño según uso", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
# %% DIBUJAR USANDO PYVIS
from pyvis.network import Network
import networkx as nx

# Crear red de Pyvis (con ancho y alto personalizados)
net = Network(height='800px', width='100%', bgcolor='#ffffff', font_color='black')
B_cleaned = nx.Graph()

for n, attrs in B.nodes(data=True):
    n_clean = str(n)
    B_cleaned.add_node(n_clean, **attrs)

# Copiar aristas, convirtiendo nodos también
for u, v in B.edges():
    B_cleaned.add_edge(str(u), str(v))
    
# Convertir el grafo de networkx a pyvis
net.from_nx(B_cleaned)

# Opcional: colorear nodos según tipo (bipartito) y dar más info al pasar el mouse
for node in net.nodes:
    tipo = B_cleaned.nodes[node['id']].get('bipartite')
    if tipo == 'user':
        node['color'] = 'blue'
        user_info = df[df['user_id'] == node['id']].dropna(subset=['user_created_at'])
        if not user_info.empty:
            created = user_info['user_created_at'].iloc[0]
            node['title'] = f"Usuario creado en: {created}"
    elif tipo == 'hashtag':
        node['color'] = 'green'
        node['title'] = f"Hashtag: {node['id']}"

# Controlar layout (force-based interactivo)
net.set_options("""
var options = {
  "layout": {
    "improvedLayout": false
  },
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -30,
      "centralGravity": 0.01,
      "springLength": 100,
      "springConstant": 0.05,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based",
    "timestep": 0.35,
    "stabilization": {
      "enabled": true,
      "iterations": 150
    }
  }
}
""")
net.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=100, spring_strength=0.05)
# Mostrar (se guarda como HTML)
net.show('red_interactiva.html')

# %%
