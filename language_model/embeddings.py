
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import zipfile

import tensorflow as tf

# where to load the corpus texts
CORPUS_ZIP = "corpus.zip"
CORPUS_TXT = "corpus.txt"

# hyper-parameters
VOCABULARY_SIZE = 50000
BATCH_SIZE = 128
NEGATIVE_SAMPLES = 64
WINDOW_SIZE = 1
NUM_SKIPS = 2
EMBEDDING_SIZE = 128

# evaluation
NEIGHBORS_K = 8


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
    samples = spares[:size]
    spares = spares[len(samples):]

    # fill up the batch with skip word samples
    buf = []
    inc = 0
    while len(samples) < size:
        buf = n_skips(corpus, window_size, num_skips)
        inc = size - len(samples)
        samples.extend(buf[:inc])

    # the batch is full; remember the spares for next time
    spares += buf[inc:]
    return samples


def print_batch(samples, words):
    for sample in samples:
        print(words[sample[0]], "|", words[sample[1]])


def print_neighbors(words, labels, neighbors):
    for i in range(0, len(neighbors)):
        neighbor_words = list()
        for j in range(0, len(neighbors[i])):
            neighbor_words.append(words[neighbors[i, j]])
        print(words[labels[i]], '->', neighbor_words)


class Embeddings(tf.layers.Layer):
    def __init__(self, vocabulary_size, embedding_size=128, **kwargs):
        super(Embeddings, self).__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embeddings = None

    def build(self, _):

        shape = [self.vocabulary_size, self.embedding_size]
        self.embeddings = self.add_variable('embeddings',
                                            shape,
                                            dtype=tf.float32,
                                            initializer=tf.random_uniform_initializer(-1.0, 1.0))

        super(Embeddings, self).build(_)

    def call(self, inputs, **kwargs):
        return super(Embeddings, self).call(self.embeddings, **kwargs)


def embeddings(vocabulary_size, embedding_size=128):
    return Embeddings(vocabulary_size, embedding_size).apply([])


class NCELoss(tf.layers.Layer):
    def __init__(self, vocabulary_size, embedding_size=128, negative_samples=64, **kwargs):
        super(NCELoss, self).__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.negative_samples = negative_samples
        self.nce_w = None
        self.nce_b = None

    def build(self, _):

        self.nce_w = self.add_variable('nce_w',
                                       [self.vocabulary_size, self.embedding_size],
                                       dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(
                                           stddev=1.0 / math.sqrt(self.embedding_size)))

        self.nce_b = self.add_variable('nce_b',
                                       [self.vocabulary_size],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())

        super(NCELoss, self).build(_)

    def call(self, inputs, **kwargs):
        inputs = super(NCELoss, self).call(inputs, **kwargs)
        input_embeddings, input_batch = inputs

        # center and context words
        batch_t = tf.transpose(input_batch)
        center_labels = batch_t[0]
        center_embeddings = tf.nn.embedding_lookup(input_embeddings, center_labels)
        context_labels = tf.expand_dims(batch_t[1], 1)

        # loss
        loss = tf.nn.nce_loss(weights=self.nce_w,
                              biases=self.nce_b,
                              labels=context_labels,
                              inputs=center_embeddings,
                              num_sampled=self.negative_samples,
                              num_classes=self.vocabulary_size)

        return tf.reduce_mean(loss)


def nce_loss(input_embeddings, input_batch, vocabulary_size,
             embedding_size=128,
             negative_samples=64):

    layer = NCELoss(vocabulary_size,
                    embedding_size=embedding_size,
                    negative_samples=negative_samples)

    return layer.apply([input_embeddings, input_batch])


class NearestNeighbors(tf.layers.Layer):
    def __init__(self, k, **kwargs):
        super(NearestNeighbors, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs, **kwargs):
        inputs = super(NearestNeighbors, self).call(inputs, **kwargs)
        embeds, labels = inputs

        # calculate cosine similarity
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeds), 1, keep_dims=True))
        normalized_embeds = embeds / norm
        normalized_examples = tf.nn.embedding_lookup(normalized_embeds, labels)
        similarity = tf.matmul(normalized_examples, normalized_embeds, transpose_b=True)

        # exclude top result (current label)
        _, top_k = tf.nn.top_k(similarity, self.k + 1)
        return top_k[:, 1:self.k + 1]


def nearest_neighbors(embeds, labels, k):
    return NearestNeighbors(k).apply([embeds, labels])


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

    graph = tf.Graph()
    with graph.as_default():
        input_batch = tf.placeholder(tf.int32, shape=[None, 2])
        embeds = embeddings(VOCABULARY_SIZE, EMBEDDING_SIZE)
        loss = nce_loss(input_embeddings=embeds,
                        input_batch=input_batch,
                        vocabulary_size=VOCABULARY_SIZE,
                        embedding_size=EMBEDDING_SIZE)
        top_k = nearest_neighbors(embeds, tf.transpose(input_batch)[0], NEIGHBORS_K)
        init = tf.global_variables_initializer()

        with tf.Session(graph=graph).as_default():

            print("initializing")
            init.run()

            print("training")
            loss_acc = 0
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
            for step in range(0, 100000):
                samples = batch(corpus, BATCH_SIZE, WINDOW_SIZE, NUM_SKIPS)
                optimizer.run({input_batch: samples})
                if step % 10000 == 0:
                    k_eval = top_k.eval({input_batch: samples[:2]})
                    print_neighbors(words, samples[:8][0], k_eval)
                loss_acc += loss.eval({input_batch: samples})
                if step % 2000 == 0:
                    print('step', step, 'avg loss', loss_acc / 2000)
                    loss_acc = 0


main()
