import pandas as pd
import webbrowser

from pyvis.network import Network

DATASET = 'tweets_with_groups_and_urls_all_without_RT'
CSV = '.csv'
data = pd.read_csv(DATASET+CSV)

net = Network(notebook = True, cdn_resources = "remote",
                bgcolor = "#'222222'",
                font_color = "white",
                height = "750px",
                width = "100%",
                # select_menu=True,
                # filter_menu=True,
)

sample = data.sample(500, random_state = 2)
sample.head(10)
sample = sample.dropna(subset=['user_id', 'lang'])
sample['user_id'] = sample['user_id'].astype(str)
sample['group'] = sample['group'].astype(str)
sample = sample.drop_duplicates(subset=['user_id', 'group'])


edges = []
user_groups = {}

for _, row in sample.iterrows():
    net.add_node(
        row['user_id'],          # ID del nodo
        label=row['user_id'],    # Texto visible
        group=row['lang']       # Esto agrupa por color automáticamente
    )



# net.add_nodes(nodes)
net.add_edges(edges)
net.show_buttons(filter_=['physics'])
net.barnes_hut()    
net.show("graph.html")

webbrowser.open("graph.html")