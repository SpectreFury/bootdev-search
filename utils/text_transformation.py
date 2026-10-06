# Transformation pipeline
# 1. Make the text lowercase

import string

translation = str.maketrans('', '', string.punctuation)

def transform_text(text: str) -> str:
    return text.lower().translate(translation)

def tokenize_text(text: str) -> list[str]:
    return text.split()
