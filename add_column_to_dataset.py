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
from classification_propaganda import classify_propaganda
from classification_emotion import classify_emotion
from classification_pysentimiento import classify_pysentimiento 



from tqdm_joblib import tqdm_joblib

def process_text(text, lang):
    # return classify_emotion(text, lang= lang)
    # return classify_propaganda(text)
    return classify_pysentimiento(text, lang= lang)



DATASET = 'tweets_with_groups_and_urls_all_without_RT_with_sentiment'

CSV = '.csv'
dataset_df = pd.read_csv(DATASET + CSV)

NEW_COLUMN = 'pyemotion'
TEXT_COLUMN = 'text'
LANG_COLUMN = 'lang'

# Keeping only eng and spa columns
ALLOWED_VALUES = ['es', 'en']
dataset_df = dataset_df[dataset_df['lang'].isin(ALLOWED_VALUES)]

with tqdm_joblib(tqdm(desc="Processing rows", total=len(dataset_df))):
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(process_text)(text, lang) for text, lang in zip(dataset_df[TEXT_COLUMN], dataset_df[LANG_COLUMN])
    )

dataset_df[NEW_COLUMN] = results
print(dataset_df)

dataset_df.to_csv(DATASET + '_with_'+NEW_COLUMN + CSV, index=0)
