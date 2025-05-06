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
DATE = '28_04'
df.to_csv('dataset_'+DATE+'.csv', index=0)

# %%
