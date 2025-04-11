# import pandas as pd

# from tqdm import tqdm
# from classification_sentiment import classify_sentiment


# def process_text(text):
#         return classify_sentiment(text)
    
# DATASET = 'tweets_with_groups_and_urls_all_without_RT'
# CSV = '.csv'
# dataset_df = pd.read_csv(DATASET+CSV)

# NEW_COLUMN = 'sentiment'
# TEXT_COLUMN = 'text'

# #Keeping only eng and spa columns
# ALLOWED_VALUES = ['es', 'en']  
# dataset_df = dataset_df[dataset_df['lang'].isin(ALLOWED_VALUES)]

# tqdm.pandas()
# dataset_df[NEW_COLUMN] = dataset_df[TEXT_COLUMN].progress_apply(lambda x: process_text(x))
# print(dataset_df)

# dataset_df.to_csv(DATASET + '_with_'+NEW_COLUMN + CSV, index=0)

##PARALEL
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from classification_sentiment import classify_sentiment
from tqdm_joblib import tqdm_joblib

def process_text(text):
    return classify_sentiment(text)


DATASET = 'tweets_with_groups_and_urls_all_without_RT'
CSV = '.csv'
dataset_df = pd.read_csv(DATASET + CSV)

NEW_COLUMN = 'sentiment'
TEXT_COLUMN = 'text'

# Keeping only eng and spa columns
ALLOWED_VALUES = ['es', 'en']
dataset_df = dataset_df[dataset_df['lang'].isin(ALLOWED_VALUES)]

def process_row(text):
    result = process_text(text)
    return result

with tqdm_joblib(tqdm(desc="Processing rows", total=len(dataset_df))):
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(process_row)(text) for text in dataset_df[TEXT_COLUMN]
    )
    
dataset_df[NEW_COLUMN] = results
print(dataset_df)

dataset_df.to_csv(DATASET + '_with_'+NEW_COLUMN + CSV, index=0)
