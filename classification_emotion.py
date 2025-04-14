from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
from scipy.special import softmax

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

MODEL_EN = f"pysentimiento/robertuito-emotion-analysis"
MODEL_SPA = f"cardiffnlp/twitter-roberta-large-emotion-latest"

# %%
tokenizer_en = AutoTokenizer.from_pretrained(MODEL_EN)
config_en = AutoConfig.from_pretrained(MODEL_EN)
model_en = AutoModelForSequenceClassification.from_pretrained(MODEL_EN)

tokenizer_es = AutoTokenizer.from_pretrained(MODEL_SPA)
config_es = AutoConfig.from_pretrained(MODEL_SPA)
model_es = AutoModelForSequenceClassification.from_pretrained(MODEL_SPA)
# config.id2label = {0: "sadness", 
#                    1: "joy",
#                    2: "love",
#                    3: "anger",
#                    4: "fear",
#                    5: "surprise",
#                    }
# %%
def classify_emotion(text: str, lang: str = "en"):
    text = preprocess(text)
    if lang =='en':
        tokenizer = tokenizer_en
        model = model_en
        config = config_en
    else:   
        tokenizer = tokenizer_es
        model = model_es
        config = config_es
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]  
    output = config.id2label[ranking[0]]
    print(text, lang,output)
    # %%
    
    return output