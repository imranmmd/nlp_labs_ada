import re
from collections import Counter
import time



class BPE:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.merges = []
        self.corpus = {}

    def to_symbols(self, word_freqs):
        return {tuple(word) + ("_",): count for word, count in word_freqs.items()}

    def get_pairs_words(self, corpus):
        pairs = Counter()

        for word, freq in corpus.items():
            for pair in zip(word, word[1:]):
                pairs[pair] += freq

        return pairs

    def merge_pair(self, corpus, pair):
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

    def _first_seen(self, corpus):
        order = {}

        for word in corpus:
            for pair in zip(word, word[1:]):
                if pair not in order:
                    order.setdefault(pair, len(order))

        return order

    def learn_bpe(self, word_freqs, num_merges):
        corpus = self.to_symbols(word_freqs)
        merges = []

        for i in range(num_merges):
            counts = self.get_pairs_words(corpus)

            if not counts:
                break

            order = self._first_seen(corpus)

            best = min(
                counts,
                key=lambda pair: (-counts[pair], order.get(pair, 10**8))
            )

            merges.append(best)
            corpus = self.merge_pair(corpus, best)

        return merges, corpus

    def apply_bpe(self, word, merges):
        word = tuple(word) + ("_",)
        symbols = word

        for left, right in merges:
            symbols = tuple(
                self.merge_pair({symbols: 1}, (left, right)).keys()
            )[0]

        return list(symbols)

    def tokenize_bpe(self, text, merges):
        tokens = []

        for word in text.split():
            tokens.extend(self.apply_bpe(word, merges))

        return tokens

    # Pipeline methods

    def fit(self, corpus):
        word_freqs = Counter()

        for sentence in corpus:
            for word in sentence.split():
                word_freqs[word] += 1

        self.merges, self.corpus = self.learn_bpe(
            word_freqs,
            self.num_merges
        )

        return self

    def encode(self, text):
        return self.tokenize_bpe(text, self.merges)

    def fit_transform(self, corpus):
        self.fit(corpus)
        return [self.encode(sentence) for sentence in corpus]

    def transform(self, corpus):
        return [self.encode(sentence) for sentence in corpus]


# ---------------- Azerbaijani stemmer ----------------

VOWELS = "aeıioöuüə"

CASES = [
    "da", "də", "dan", "dən", "nın", "nin", "nun", "nün",
    "nı", "ni", "nu", "nü", "na", "nə", "nda", "ndə", "ndan", "ndən",
    "ın", "in", "un", "ün", "ya", "yə", "la", "lə", "ca", "cə",
]

POSSESSIVE = [
    "ım", "im", "um", "üm", "ımız", "imiz", "umuz", "ümüz",
    "ınız", "iniz", "unuz", "ünüz", "sı", "si", "su", "sü", "ları", "ləri",
]

PLURAL = ["lar", "lər"]

DERIVATIONAL = [
    "çı", "çi", "çu", "çü", "lıq", "lik", "luq", "lük",
    "sız", "siz", "suz", "süz", "lı", "li", "lu", "lü",
    "çılıq", "çilik", "çuluq", "çülük",
]

VERBAL = [
    "maq", "mək", "dı", "di", "du", "dü", "mış", "miş", "muş", "müş",
    "acaq", "əcək", "ır", "ir", "ur", "ür", "ıb", "ib", "ub", "üb",
]

SINGLE = ["a", "ə", "ı", "i", "u", "ü"]

PREFIXES = ["qeyri", "anti", "bi", "na"]


def make_regex(suffixes):
    suffixes = sorted(suffixes, key=len, reverse=True)
    return re.compile("(" + "|".join(suffixes) + ")$")


STAGES = [
    make_regex(CASES),
    make_regex(POSSESSIVE),
    make_regex(PLURAL),
    make_regex(DERIVATIONAL),
    make_regex(VERBAL),
    make_regex(SINGLE),
]

PREFIX_RE = re.compile("^(" + "|".join(PREFIXES) + ")")

ALL_SUFFIXES = set(CASES + POSSESSIVE + PLURAL + DERIVATIONAL + VERBAL)


def lower_az(word):
    word = word.replace("I", "ı").replace("İ", "i")
    return word.lower()


def has_vowel(word):
    return any(ch in VOWELS for ch in word)


class AzStemmer:
    def __init__(self, num_merges=200):
        self.bpe = BPE(num_merges=num_merges)

    def fit(self, sentences):
        clean = []
        for s in sentences:
            clean.append(" ".join(lower_az(w) for w in re.findall(r"\w+", s)))
        self.bpe.fit(clean)
        return self

    def remove_prefix(self, word):
        m = PREFIX_RE.match(word)
        if m and len(word) - len(m.group(1)) >= 5:
            return word[m.end():]
        return word

    def strip_bpe_suffixes(self, word):
        tokens = self.bpe.encode(word)
        tokens[-1] = tokens[-1].rstrip("_")
        tokens = [t for t in tokens if t]

        while len(tokens) > 1 and len(tokens[-1]) >= 3 and tokens[-1] in ALL_SUFFIXES:
            stem = "".join(tokens[:-1])
            if len(stem) < 3 or not has_vowel(stem):
                break
            tokens.pop()

        return "".join(tokens)

    def strip_regex_suffixes(self, word):
        for regex in STAGES:
            m = regex.search(word)
            if not m:
                continue
            stem = word[:-len(m.group(1))]
            min_len = 4 if regex is STAGES[-1] else 2
            if len(stem) >= min_len and has_vowel(stem):
                word = stem
        return word

    def fix_alternation(self, word):
        if word.endswith("ğ"):
            word = word[:-1] + "q"
        elif word.endswith("əy"):
            word = word[:-2] + "ək"
        return word

    def stem(self, word):
        word = lower_az(word)

        if len(word) <= 3:
            return word

        word = self.remove_prefix(word)
        word = self.strip_bpe_suffixes(word)
        word = self.strip_regex_suffixes(word)
        word = self.fix_alternation(word)
        return word

    def stem_text(self, text):
        return [self.stem(w) for w in re.findall(r"\w+", text)]



ABBR = {"məs.", "prof.", "dos.", "akad.", "şək.", "səh.", "bax.", "cəm.", "sm.", "hk.", "bənd.", "m.ö.", "e.ə."}

def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r'([.!?…]+["»”)]*)\s+', text)

    pieces = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)]
    if len(parts) % 2 == 1 and parts[-1]:
        pieces.append(parts[-1])

    sentences = []
    for piece in pieces:
        if sentences:
            last_word = sentences[-1].split()[-1].lower()
            is_abbr = last_word in ABBR or (len(last_word) == 2 and last_word.endswith("."))
            starts_lower = piece[0].islower()
            if is_abbr or starts_lower:
                sentences[-1] += " " + piece
                continue
        sentences.append(piece)

    return sentences

text = ""

print("Reading and normalizing Azerbaijani text from file...")
start_time = time.time()
with open("data/azerbaijan_school_textbooks.txt", "r", encoding="utf-8") as file:
    text = file.read()
end_time = time.time()
print(f"Time taken for reading and normalizing: {end_time - start_time:.2f} seconds")


corpus = split_sentences(text)
print(corpus[:1000])


stemmer = AzStemmer(num_merges=1_000).fit(corpus)
test_words = [
    "kitablar", "kitabları", "kitabımızda", "məktəblilərdən",
    "evlərimizdən", "müəllimlərin", "gözəllik", "işçilər",
    "azadlıq", "dostluq", "oxumaq", "yazıçılar", "uşaqlar",
    "qonağı", "çörəyi", "masasında", "Ağıllılar", "çayxanamız", "uşaqları", "bildirməkləri"
]

for w in test_words:
    print(f"{w:18} -> {stemmer.stem(w)}")