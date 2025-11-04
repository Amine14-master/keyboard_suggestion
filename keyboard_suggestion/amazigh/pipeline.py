import json
import re
from collections import defaultdict, Counter
from math import log

# ======== (1) Nettoyage du texte ========

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)           # Balises HTML
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)   # Templates wiki
    text = re.sub(r"\[\[.*?\]\]", " ", text)       # Liens internes
    text = re.sub(r"\[https?:\/\/[^\]]*\]", " ", text)  # Liens externes
    text = re.sub(r"[^0-9A-Za-z\u0600-\u06FF\u2D30-\u2D7F’'ʿ_\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ======== (2) Tokenisation ========

def tokenize(text):
    text = text.lower()
    tokens = text.split()
    return tokens

# ======== (3) Construction des N-grammes ========

def build_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i + n]))
    return ngrams

# ======== (4) Calcul des probabilités ========

def calculate_probabilities(ngram_counts, lower_counts):
    probabilities = {}

    vocab_size = len(lower_counts)

    for ngram, count in ngram_counts.items():
        prefix = ngram[:-1]

        prefix_count = lower_counts[prefix] if prefix in lower_counts else 0

        prob = (count + 1) / (prefix_count + vocab_size)

        probabilities[ngram] = -log(prob)   # log-proba (plus stable)

    return probabilities

# ======== (5) Pipeline complet ========

def process_corpus(input_file, output_prefix):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)
    tokens = tokenize(text)

    print("Total tokens:", len(tokens))

    # Count 1→5 grams
    counts = {n: Counter() for n in range(1, 6)}

    for n in range(1, 6):
        grams = build_ngrams(tokens, n)
        counts[n].update(grams)

    # Probabilités
    probabilities = {}

    for n in range(2, 6):
        probabilities[n] = calculate_probabilities(
            counts[n],
            {k: v for k, v in counts[n - 1].items()}
        )

    # Sauvegarde JSON
    for n in range(1, 6):
        out_path = f"{output_prefix}_{n}gram_counts.json"
        json.dump({" ".join(k): v for k, v in counts[n].items()},
                  open(out_path, "w", encoding="utf-8"), ensure_ascii=False)

    for n in range(2, 6):
        out_path = f"{output_prefix}_{n}gram_logproba.json"
        json.dump({" ".join(k): v for k, v in probabilities[n].items()},
                  open(out_path, "w", encoding="utf-8"), ensure_ascii=False)

    print(" Terminé – fichiers générés :")
    print("    - counts 1→5-gram")
    print("    - log-probabilités 2→5-gram")


# ======== EXECUTION ========

process_corpus("kabyle_corpus.txt", "kabyle")
