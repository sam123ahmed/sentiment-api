import re

def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("`", "'")
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text