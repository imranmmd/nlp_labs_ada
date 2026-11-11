from bpe_learner_segmenter import learn_bpe, tokenize_bpe
from collections import Counter
import time

def normalize_text_azerbaijani(text):
    # Normalize Azerbaijani text by converting to lowercase and removing punctuation
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

text = ""

print("Reading and normalizing Azerbaijani text from file...")
start_time = time.time()
with open("data/azerbaijan_school_textbooks.txt", "r", encoding="utf-8") as file:
    text = file.read()
    text = normalize_text_azerbaijani(text)
end_time = time.time()
print(f"Time taken for reading and normalizing: {end_time - start_time:.2f} seconds")

print("Learning BPE...")
start_time = time.time()
merges, corpus = learn_bpe(Counter(text.split()), 1000)
end_time = time.time()
print(f"Time taken for BPE learning: {end_time - start_time:.2f} seconds")

print("Tokenizing text using learned BPE merges...")
start_time = time.time()
tokens = tokenize_bpe(normalize_text_azerbaijani("Azərbaycanın inkişafı tarixi amillərdən asılııq yaradacaqdır"), merges)
end_time = time.time()
print(f"Time taken for tokenization: {end_time - start_time:.2f} seconds")

longest_tokens = sorted(tokens, key=len, reverse=True)[:50]
print("Longest 50 tokens:", longest_tokens)