from itertools import combinations
import pandas as pd
from pyvis.network import Network

DATASET = 'tweets_with_groups_and_urls_all_without_RT'
CSV = '.csv'
data = pd.read_csv(DATASET+CSV)

net = Network(notebook = True, cdn_resources = "remote",
                bgcolor = "#222222",
                font_color = "white",
                height = "750px",
                width = "100%",
                # select_menu=True,
                # filter_menu=True,
)

sample = data.sample(1000, random_state = 1)
sample.head(10)
sample = sample.dropna(subset=['user_id', 'group'])  # Quitamos filas incompletas
sample['user_id'] = sample['user_id'].astype(str)


# nodes = list(set([*sample.person1,*sample.person2]))
nodes = sample['user_id'].dropna().astype(str).unique().tolist()
edges = []
user_groups = {}

# Agrupamos por 'group' y construimos edges + mapeamos user_id → group
for _, group_df in sample.groupby('group'):
    users = group_df['user_id'].unique()
    for u in users:
        user_groups[u] = group_df['group'].iloc[0]  # le asignamos grupo
    for u1, u2 in combinations(users, 2):
        edges.append((u1, u2))
# Extraer todos los nodos únicos que aparecen en los edges
nodes = set()
for u1, u2 in edges:
    nodes.add(u1)
    nodes.add(u2)
    
# Agrupamos por 'group' y conectamos cada par de usuarios dentro de ese grupo
for _, group_df in sample.groupby('group'):
    users = group_df['user_id'].dropna().astype(str).unique()
    for u1, u2 in combinations(users, 2):
        edges.append((u1, u2))
# Agregar nodos con grupo
for node in nodes:
    group = user_groups.get(node, 'unknown')
    net.add_node(node, label=node, group=group)

net.add_nodes(nodes)
net.add_edges(edges)
net.show_buttons(filter_=['physics'])
net.barnes_hut()    
net.show("graph.html")

import webbrowser
webbrowser.open("graph.html")