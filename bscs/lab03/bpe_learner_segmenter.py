
from collections import Counter

def to_symbols(word_freqs):
    return {tuple(word) + ("_",): count for word, count in word_freqs.items()}

def get_pairs_words(corpus):
    pairs = Counter()
    for word, freq in corpus.items():
        for pair in zip(word, word[1:]):
            pairs[pair] += freq
    return pairs

def merge_pair(corpus, pair):
    new_corpus = {}
    bigram = ''.join(pair)
    for word, freq in corpus.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(bigram)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus[tuple(new_word)] = freq
    return new_corpus

def _first_seen(corpus):
    order = {}
    for word in corpus:
        for pair in zip(word, word[1:]):
            if pair not in order:
                order.setdefault(pair, len(order))
    return order

def learn_bpe(word_freqs, num_merges):
    corpus = to_symbols(word_freqs)
    merges = []
    for i in range(num_merges):
        counts = get_pairs_words(corpus)
        if not counts:
            break
        order = _first_seen(corpus)
        best = min(counts, key=lambda pair: (-counts[pair], order.get(pair, 10**8)))
        merges.append(best)
        corpus = merge_pair(corpus, best)
    return merges, corpus

def apply_bpe(word, merges):
    word = tuple(word) + ("_",)
    symbols = word
    for left, right in merges:
        symbols = tuple(merge_pair({symbols: 1}, (left, right)).keys())[0]
    return list(symbols)

def tokenize_bpe(text, merges):
    tokens = []
    for word in text.split():
        tokens.extend(apply_bpe(word, merges))
    return tokens
