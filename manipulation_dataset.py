import pandas as pd

NAME = 'tweets_with_groups_and_urls_all_without_RT_with_sentiment'
df = pd.read_csv(NAME + '.csv')

user_id_counts = df['user_id'].value_counts()

NEW_COLUMN = 'user_id_count'

df[NEW_COLUMN] = df['user_id'].map(user_id_counts)

df.to_csv(NAME + '_with_'+NEW_COLUMN +'.csv', index=0)
