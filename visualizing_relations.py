#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
DATE = '28_04'
NAME = 'dataset_' + DATE
df = pd.read_csv(NAME+'.csv')

#%%
df['group'] = df['group'].apply(lambda x: eval(x)[0] if isinstance(x, str) else x)
#Sentiment - group relativerelations
relative_df = df.groupby(['group', 'sentiment']).size().reset_index(name='count')
total_by_group = relative_df.groupby('group')['count'].transform('sum')
relative_df['percentage'] = relative_df['count'] / total_by_group * 100

plt.figure(figsize=(10, 6))
sns.barplot(data=relative_df, x='group', y='percentage', hue='sentiment')
plt.title('Proporción de sentimientos por grupo')
plt.xlabel('Grupo')
plt.ylabel('Porcentaje (%)')
plt.xticks(rotation=45)
plt.legend(title='Sentimiento')
plt.tight_layout()
plt.show()

#%%
df['user_created_at'] = pd.to_datetime(df['user_created_at'])
df_sorted = df.sort_values(by='user_created_at')
# Grafica 'user_created_at' vs 'sentiment'
plt.figure(figsize=(18, 8))
plt.scatter(df_sorted['user_created_at'], df_sorted['sentiment'], alpha=0.3, s=10)
plt.title('Relación entre user_created_at y sentiment')
plt.xlabel('Fecha de creación del usuario')
plt.ylabel('Sentimiento')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %%
df_exploded = df.explode('hashtags')

top_n = 20  # Número de hashtags más comunes que quieres mostrar
top_hashtags = df_exploded['hashtags'].value_counts().head(top_n).index
df_filtered = df_exploded[df_exploded['hashtags'].isin(top_hashtags)]

counts = df_filtered.groupby(['hashtags', 'sentiment']).size().reset_index(name='count')

# Segundo: sumamos por hashtag
total_counts = counts.groupby('hashtags')['count'].transform('sum')

# Tercero: sacamos el porcentaje
counts['percentage'] = counts['count'] / total_counts

# Ahora graficamos
plt.figure(figsize=(20, 10))
sns.barplot(data=counts, x='hashtags', y='percentage', hue='sentiment')

plt.title('Distribución relativa de sentimiento por hashtag')
plt.xlabel('Hashtag')
plt.ylabel('Proporción')
plt.xticks(rotation=90)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()
# %%

# Agrupar por cantidad de tweets y sentimiento
grouped = df.groupby(['user_id_count', 'sentiment']).size().unstack(fill_value=0)

# Convertir a porcentaje
grouped_percentage = grouped.div(grouped.sum(axis=1), axis=0)

# Graficar como stacked barplot
grouped_percentage.plot(kind='bar', stacked=True, figsize=(20, 8), colormap='Set2')

plt.title('Distribución relativa de sentimiento según número de tweets del usuario')
plt.xlabel('Cantidad de tweets del usuario (user_id_count)')
plt.ylabel('Proporción de sentimientos')
plt.xticks(rotation=90)
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12,6))
sns.histplot(df['text_length_ratio'], bins=50, kde=True)
plt.title('Distribución del ratio de longitud de texto preprocesado vs original')
plt.xlabel('Proporción de caracteres conservados')
plt.ylabel('Número de filas (textos)')
plt.tight_layout()
plt.show()


print("Promedio de proporción conservada:", df['text_length_ratio'].mean())
print("Mediana de proporción conservada:", df['text_length_ratio'].median())
print("Porcentaje de textos que pierden más del 50%:", (df['text_length_ratio'] < 0.5).mean() * 100)


# %%
