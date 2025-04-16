import pandas as pd
import webbrowser
import networkx as nx

from pyvis.network import Network

DATASET = 'tweets_with_groups_and_urls_all_without_RT_with_sentiment_with_emotion'
CSV = '.csv'
data = pd.read_csv(DATASET+CSV)
GROUP = 'sentiment'
VALUE = 'user_created_at'


net = Network(notebook = True,
                bgcolor = "#'222222'",
                font_color = "white",
                height = "750px",
                width = "100%",
                cdn_resources='remote',
                # select_menu=True,
                filter_menu=True,
)

sample = data.sample(1000, random_state = 3)
sample.head(10)
sample = sample.dropna(subset=['user_id', GROUP])
sample['user_id'] = sample['user_id'].astype(str)
sample['value'] = pd.to_datetime(sample['user_created_at']).astype(int) // 10**9
sample = sample.drop_duplicates(subset=['user_id', GROUP])

sample['value'] = pd.to_datetime(sample['value'])

# Establecer la fecha base (puedes elegir cualquier fecha o la más temprana)
base_date = sample['value'].min()
# Calcular la diferencia en segundos con respecto a la fecha base
sample['seconds_diff'] = (sample['value'] - base_date).dt.total_seconds()

# Convertir la diferencia de segundos a días, si lo prefieres
sample['days_diff'] = sample['seconds_diff'] / (60 * 60 * 24)  # Convierte segundos a días

max_size = 100  # Tamaño máximo de los nodos
min_size = 10  # Tamaño mínimo de los nodos

sample['size'] = (sample['days_diff'] / sample['days_diff'].max()) * (max_size - min_size) + min_size

for _, row in sample.iterrows():
    group_color = {'negative': 'red', 'neutral': 'blue', 'positive': 'green'}.get(row[GROUP], 'gray')
    print(group_color)
    net.add_node(row['id'], label=row['id'], group=row[GROUP], 
                 title=f"Id: {row['id']}, Fecha: {row['user_created_at']}", color=group_color, size=row['size'])

net.show_buttons(filter_=['physics'])
net.show('graph.html')
webbrowser.open('graph.html')


import matplotlib.pyplot as plt
plt.figure()
x = sample['user_created_at']
y = sample['sentiment']
plt.ylabel('Creation Date')
plt.bar(x, y, color='blue')
