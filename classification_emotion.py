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

MODEL_MULTI="Toshifumi/bert-base-multilingual-cased-finetuned-emotion"
MODEL_MULTI2 = "agustinst1990/distilbert-base-multilingual-cased-finetuned-emotion"

MODEL = MODEL_MULTI2
# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)
# %%
# dictionary: MODEL_MULTI
config.id2label = {0: "sadness", 
                   1: "joy",
                   2: "love",
                   3: "anger",
                   4: "fear",
                   5: "surprise",
                   }
# %%

def classify(text: str):
    # %%
    text = preprocess(text)
    print(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]  
    output = config.id2label[ranking[0]]
    print(output)
    # %%
    
    return output