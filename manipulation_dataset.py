import pandas as pd

df = pd.read_csv('tweets_with_groups_and_urls_all_without_RT_with_sentiment.csv')

user_id_counts = df['user_id'].value_counts()

# Crear una nueva columna en el dataframe con ese conteo
df['user_id_count'] = df['user_id'].map(user_id_counts)
