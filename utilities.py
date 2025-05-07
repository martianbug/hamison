import ast



# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def normalize_column(value):
    if isinstance(value, list):
        return [str(x).upper().strip() for x in value if x.strip()]
    elif isinstance(value, str):
        return [value.upper().strip()]
    else:
        return []

def string_to_list(string):
    return ast.literal_eval(string)
