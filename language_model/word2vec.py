
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import zipfile

# where to load the corpus texts
CORPUS_ZIP = "corpus.zip"
CORPUS_TXT = "corpus.txt"

# hyper-parameters
VOCABULARY_SIZE = 50000
BATCH_SIZE = 64
WINDOW_SIZE = 1
NUM_SKIPS = 2


def load_corpus():
    with zipfile.ZipFile(CORPUS_ZIP) as f:
        return f.read(CORPUS_TXT).split()


def build_vocabulary(text_corpus):

    # occurrence counts of vocabulary words
    counts = collections.Counter(text_corpus).most_common(VOCABULARY_SIZE - 1)

    # build a dictionary that maps words to codes
    # (rare words all share the same code)
    codes = dict(UNK=0)
    for word, _ in counts:
        codes[word] = len(codes)

    # also map codes to words (reverse dictionary)
    words = dict(zip(codes.values(), codes.keys()))

    # process the textual corpus into codes
    corpus = list()
    unk_count = 0
    unk_code = codes.get('UNK')
    for word in text_corpus:
        code = codes.get(word, unk_code)
        if code == unk_code:
            unk_count += 1
        corpus.append(code)

    # insert unknown word count in ordered position
    unk_index = VOCABULARY_SIZE - 1
    for i, count in enumerate(counts):
        if unk_count > count[1]:
            unk_index = i
            break
    counts.insert(unk_index, ['UNK', unk_count])

    return counts, codes, words, corpus


corpus_i = 0


def n_skips(corpus, w, max_n):
    global corpus_i

    corpus_len = len(corpus)

    # where in the corpus to find context words
    # (stay within bounds)
    a = corpus_i - w if corpus_i >= w else 0
    b = corpus_i + w if corpus_i < corpus_len - w else corpus_len - 1

    # extract the center and context words
    center = corpus[corpus_i]
    context = corpus[a:corpus_i] + corpus[corpus_i + 1:b + 1]

    # pick up to max_n skip words from the context
    n = min(max_n, len(context))
    picks = random.sample(context, n)

    # each sample consists of the center word and one random context word
    skips = []
    for pick in picks:
        skips.append([center, pick])

    # step forward thru the corpus
    corpus_i += 1
    if corpus_i >= corpus_len:
        corpus_i = 0

    return skips


spares = []


def batch(corpus, size, window_size, num_skips):
    global spares

    # pick up left-over samples from previous batch
    # (but don't overfill if the batch size is small)
    samples = spares[:min(len(spares), size)]
    spares = spares[len(samples):]

    # fill up the batch with skip word samples
    buf = []
    inc = 0
    while len(samples) < size:
        buf = n_skips(corpus, window_size, num_skips)
        inc = min(len(buf), size - len(samples))
        samples.extend(buf[:inc])

    # the batch is full; remember the spares for next time
    spares += buf[inc:]
    return samples


def print_batch(samples, words):
    for sample in samples:
        print(words[sample[0]], "|", words[sample[1]])


def main():
    text_corpus = load_corpus()
    print("len(text_corpus) =", len(text_corpus))

    counts, codes, words, corpus = build_vocabulary(text_corpus)
    print("len(counts) =", len(counts))
    print("len(codes) =", len(codes))
    print("len(words) =", len(words))
    print("len(corpus) =", len(corpus))

    # free memory
    del text_corpus

    samples = batch(corpus, BATCH_SIZE, WINDOW_SIZE, NUM_SKIPS)
    print_batch(samples[:8], words)


main()
