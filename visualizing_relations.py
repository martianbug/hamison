import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('tweets_with_groups_and_urls_all_without_RT_with_sentiment.csv')
df['group'] = df['group'].apply(lambda x: eval(x)[0] if isinstance(x, str) else x)

# Calcular proporciones de cada sentimiento por grupo
relative_df = df.groupby(['group', 'sentiment']).size().reset_index(name='count')
total_by_group = relative_df.groupby('group')['count'].transform('sum')
relative_df['percentage'] = relative_df['count'] / total_by_group * 100

# Crear gráfico de barras apiladas con proporciones
plt.figure(figsize=(10, 6))
sns.barplot(data=relative_df, x='group', y='percentage', hue='sentiment')
plt.title('Proporción de sentimientos por grupo')
plt.xlabel('Grupo')
plt.ylabel('Porcentaje (%)')
plt.xticks(rotation=45)
plt.legend(title='Sentimiento')
plt.tight_layout()
plt.show()