from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty

import collections
import numpy as np
import random
import zipfile


class Corpus:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractproperty
    def vocab_size(self):
        pass

    @abstractproperty
    def counts(self):
        pass

    @abstractproperty
    def est_progress(self):
        pass

    @abstractmethod
    def rewind(self):
        pass

    @abstractmethod
    def next_label(self):
        pass

    @abstractmethod
    def to_word(self, label):
        pass


class ZipTxtCorpus(Corpus):

    def __init__(self, zip_file, txt_file, vocab_size):
        corpus_text = self._load(zip_file, txt_file)
        counts, labels, words, corpus = self._build(corpus_text, vocab_size)
        del corpus_text
        self._counts = counts
        self._labels = labels
        self._words = words
        self._corpus = corpus
        self._index = 0

    @staticmethod
    def _load(zip_file, txt_file):
        with zipfile.ZipFile(zip_file) as f:
            return f.read(txt_file).split()

    @staticmethod
    def _build(text_corpus, vocab_size):

        # occurrence counts of vocabulary words
        counts = collections.Counter(text_corpus).most_common(vocab_size - 1)

        # build a dictionary that maps words to labels
        # (rare words all share the same label)
        labels = dict(UNK=0)
        for word, _ in counts:
            labels[word] = len(labels)

        # also map labels to words (reverse dictionary)
        words = dict(zip(labels.values(), labels.keys()))

        # process the textual lm_corpus into labels
        corpus = list()
        unk_count = 0
        unk_label = labels.get('UNK')
        for word in text_corpus:
            label = labels.get(word, unk_label)
            if label == unk_label:
                unk_count += 1
            corpus.append(label)

        # insert unknown word count in ordered position
        unk_index = vocab_size - 1
        for i, count in enumerate(counts):
            if unk_count > count[1]:
                unk_index = i
                break
        counts.insert(unk_index, ['UNK', unk_count])

        return counts, labels, words, corpus

    @property
    def vocab_size(self):
        return len(self._counts)

    @property
    def counts(self):
        return self._counts

    @property
    def est_progress(self):
        return self._index / len(self._corpus)

    def rewind(self):
        self._index = 0

    def next_label(self):
        if self._index >= len(self._corpus):
            return None
        label = self._corpus[self._index]
        self._index += 1
        return label

    def to_word(self, label):
        return self._words[label]


class NSkipSampler:
    def __init__(self, corpus, window_size, max_n):
        self._corpus = corpus
        self._window_size = window_size
        self._max_n = max_n
        self._buf = [None] * (2 * window_size + 1)

    def sample(self):

        ws = self._window_size
        buf = self._buf

        # shift buffer to the left
        for i in range(len(buf) - 1):
            buf[i] = buf[i + 1]
        buf[len(buf) - 1] = None

        # fill the buffer with the local context
        for i in range(ws, len(buf)):
            if buf[i] is not None:
                continue
            buf[i] = self._corpus.next_label()
            if buf[i] is None:
                break

        # extract center word (None indicates end of lm_corpus)
        center = buf[ws]
        if center is None:
            return []

        # extract the context words
        context = []
        for i in range(0, len(buf)):
            if i == ws:
                continue
            if buf[i] is not None:
                context.append(buf[i])

        # pick up to max_n skip words from the context
        n = min(self._max_n, len(context))
        picks = random.sample(context, n)

        # each sample consists of the center word and one random context word
        n_skips = []
        for pick in picks:
            n_skips.append([center, pick])

        return n_skips


class Batchifier():
    def __init__(self, sampler):
        self._sampler = sampler
        self._spares = []

    def batch(self, size):

        # pick up left-over samples from previous batch
        # (but don't overfill if the batch size is small)
        samples = self._spares[:size]
        self._spares = self._spares[len(samples):]

        # fill up the batch with skip word samples
        buf = ['dummy']
        inc = 0
        while len(samples) < size and buf:
            buf = self._sampler.sample()
            inc = size - len(samples)
            samples.extend(buf[:inc])

        # the batch is full; remember the spares for next time
        self._spares += buf[inc:]
        return np.array(samples)
