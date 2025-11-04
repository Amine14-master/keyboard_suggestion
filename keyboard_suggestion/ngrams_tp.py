# Construire les n-grammes et calculer les probabilités
import json
import os
import re
from collections import defaultdict
from nltk import FreqDist
from nltk.util import bigrams, trigrams, ngrams
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

def simple_tokenize(text):
    """Simple tokenization that splits on whitespace and punctuation.
    Works for English, French, and Arabic text.
    """
    # Convert to lowercase
    text = text.lower()
    # Split on whitespace first
    words = text.split()
    tokens = []
    for word in words:
        # Split word and punctuation, but keep both
        # This regex finds sequences of word characters and sequences of non-word characters
        parts = re.findall(r'\w+|[^\w\s]', word)
        tokens.extend(parts)
    return [t for t in tokens if t.strip()]

# File paths for the three sentence files
FILE_PATHS = {
    'arabic': 'extracted/arabic/ara-dz_newscrawl-OSIAN_2018_10K/ara-dz_newscrawl-OSIAN_2018_10K-sentences.txt',
    'english': 'extracted/english/eng_news_2024_10K/eng_news_2024_10K-sentences.txt',
    'french': 'extracted/french/fra_news_2024_10K/fra_news_2024_10K-sentences.txt'
}

def read_sentences_file(file_path):
    """Read sentences from file, skipping line numbers."""
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Skip line number and tab, get the sentence
                    parts = line.split('\t', 1)
                    if len(parts) > 1:
                        sentences.append(parts[1])
                    else:
                        sentences.append(line)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return []
    return sentences

def process_text_to_tokens(sentences):
    """Convert sentences to tokens."""
    tokens = []
    for sentence in sentences:
        try:
            # Use simple tokenization instead of NLTK's word_tokenize
            sentence_tokens = simple_tokenize(sentence)
            tokens.extend(sentence_tokens)
        except Exception as e:
            print(f"Error tokenizing sentence: {e}")
            continue
    return tokens

def calculate_probabilities(ngram_freq, prev_ngram_freq):
    """Calculate conditional probabilities for n-grams.
    For n-gram (w1, w2, ..., wn), calculates P(wn | w1, w2, ..., wn-1)
    """
    probs = {}
    for ngram, count in ngram_freq.items():
        if len(ngram) > 1:
            # Get the previous n-gram (all but the last word)
            prev_ngram = ngram[:-1]
            # Get count of previous n-gram
            prev_count = prev_ngram_freq.get(prev_ngram, 0)
            if prev_count > 0:
                probs[ngram] = count / prev_count
            else:
                probs[ngram] = 0.0
        else:
            # For unigrams, probability is just frequency (handled separately)
            total = sum(ngram_freq.values())
            if total > 0:
                probs[ngram] = count / total
            else:
                probs[ngram] = 0.0
    return probs

def ngram_to_string(ngram):
    """Convert n-gram tuple to string for JSON serialization."""
    return ' '.join(ngram)

def process_language(lang, file_path):
    """Process a language file and generate all n-grams and probabilities."""
    print(f"\nProcessing {lang}...")
    
    # Read sentences
    sentences = read_sentences_file(file_path)
    if not sentences:
        print(f"No sentences found for {lang}")
        return
    
    print(f"Read {len(sentences)} sentences")
    
    # Tokenize
    tokens = process_text_to_tokens(sentences)
    print(f"Generated {len(tokens)} tokens")
    
    # Create vocabulary
    vocab = sorted(set(tokens))
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    vocab_rev = {idx: word for idx, word in enumerate(vocab)}
    
    # Create output directory
    output_dir = f'ngrams_tp/{lang}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate n-grams (1 to 5)
    ngram_freqs = {}
    ngram_probs = {}
    
    # Unigrams (1-grams)
    print("Generating unigrams...")
    unigrams = FreqDist(tokens)
    ngram_freqs[1] = {ngram_to_string((word,)): count for word, count in unigrams.items()}
    total_unigrams = sum(unigrams.values())
    ngram_probs[1] = {ngram_to_string((word,)): count / total_unigrams if total_unigrams > 0 else 0.0 
                     for word, count in unigrams.items()}
    
    # Bigrams (2-grams)
    print("Generating bigrams...")
    bigram_list = list(bigrams(tokens))
    bigram_freq = FreqDist(bigram_list)
    ngram_freqs[2] = {ngram_to_string(bigram): count for bigram, count in bigram_freq.items()}
    # For bigrams, calculate P(w2|w1) = count(w1,w2) / count(w1)
    bigram_probs = {}
    for bigram, count in bigram_freq.items():
        w1 = bigram[0]
        w1_count = unigrams.get(w1, 0)
        if w1_count > 0:
            bigram_probs[bigram] = count / w1_count
        else:
            bigram_probs[bigram] = 0.0
    ngram_probs[2] = {ngram_to_string(ngram): prob for ngram, prob in bigram_probs.items()}
    
    # Trigrams (3-grams)
    print("Generating trigrams...")
    trigram_list = list(trigrams(tokens))
    trigram_freq = FreqDist(trigram_list)
    ngram_freqs[3] = {ngram_to_string(trigram): count for trigram, count in trigram_freq.items()}
    ngram_probs[3] = calculate_probabilities(trigram_freq, bigram_freq)
    ngram_probs[3] = {ngram_to_string(ngram): prob for ngram, prob in ngram_probs[3].items()}
    
    # Quadrigrams (4-grams)
    print("Generating quadrigrams...")
    quadrigram_list = list(ngrams(tokens, 4))
    quadrigram_freq = FreqDist(quadrigram_list)
    ngram_freqs[4] = {ngram_to_string(quadrigram): count for quadrigram, count in quadrigram_freq.items()}
    ngram_probs[4] = calculate_probabilities(quadrigram_freq, trigram_freq)
    ngram_probs[4] = {ngram_to_string(ngram): prob for ngram, prob in ngram_probs[4].items()}
    
    # Pentagrams (5-grams)
    print("Generating pentagrams...")
    pentagram_list = list(ngrams(tokens, 5))
    pentagram_freq = FreqDist(pentagram_list)
    ngram_freqs[5] = {ngram_to_string(pentagram): count for pentagram, count in pentagram_freq.items()}
    ngram_probs[5] = calculate_probabilities(pentagram_freq, quadrigram_freq)
    ngram_probs[5] = {ngram_to_string(ngram): prob for ngram, prob in ngram_probs[5].items()}
    
    # Save all n-gram frequencies
    print("Saving n-gram frequencies...")
    for n in range(1, 6):
        with open(f'{output_dir}/{n}gram.json', 'w', encoding='utf-8') as f:
            json.dump(ngram_freqs[n], f, ensure_ascii=False, indent=2)
    
    # Save all probabilities
    print("Saving probabilities...")
    for n in range(2, 6):
        with open(f'{output_dir}/prob_{n}gram.json', 'w', encoding='utf-8') as f:
            json.dump(ngram_probs[n], f, ensure_ascii=False, indent=2)
    
    # Save vocabulary
    print("Saving vocabulary...")
    with open(f'{output_dir}/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    with open(f'{output_dir}/vocab_rev.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_rev, f, ensure_ascii=False, indent=2)
    
    # Save frequency summary
    freq_summary = {
        'unigrams': len(unigrams),
        'bigrams': len(bigram_freq),
        'trigrams': len(trigram_freq),
        'quadrigrams': len(quadrigram_freq),
        'pentagrams': len(pentagram_freq),
        'total_tokens': len(tokens),
        'vocabulary_size': len(vocab)
    }
    with open(f'{output_dir}/freq.json', 'w', encoding='utf-8') as f:
        json.dump(freq_summary, f, ensure_ascii=False, indent=2)
    
    print(f"Completed processing {lang}")
    print(f"  - Vocabulary size: {len(vocab)}")
    print(f"  - Unigrams: {len(unigrams)}")
    print(f"  - Bigrams: {len(bigram_freq)}")
    print(f"  - Trigrams: {len(trigram_freq)}")
    print(f"  - Quadrigrams: {len(quadrigram_freq)}")
    print(f"  - Pentagrams: {len(pentagram_freq)}")

# Process all three languages
if __name__ == '__main__':
    for lang, file_path in FILE_PATHS.items():
        if os.path.exists(file_path):
            process_language(lang, file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    print("\nAll processing complete!")
