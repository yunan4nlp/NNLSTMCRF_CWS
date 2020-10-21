from collections import Counter
import numpy as np
import re

class Vocab(object):
    PAD, UNK = 0, 1
    def __init__(self, word_counter, char_counter, bichar_counter, min_occur_count = 2):
        self._id2word = ['<pad>', '<unk>']
        self._wordid2freq = [10000,  10000]

        self._id2char = ['<pad>', '<unk>']
        self._id2bichar = ['<pad>', '<unk>']

        #self._id2extchar = ['<pad>', '<unk>']
        #self._id2extbichar = ['<pad>', '<unk>']

        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        for char, count in char_counter.most_common():
            self._id2char.append(char)

        for bichar, count in bichar_counter.most_common():
            self._id2bichar.append(bichar)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._char2id = reverse(self._id2char)
        if len(self._char2id) != len(self._id2char):
            print("serious bug: chars dumplicated, please check!")

        self._bichar2id = reverse(self._id2bichar)
        if len(self._bichar2id) != len(self._id2bichar):
            print("serious bug: bichars dumplicated, please check!")

        print("Vocab info: #char %d, #bichar %d" % (self.char_size, self.bichar_size))

    def create_label(self, train_data):
        label_counter = Counter()
        for inst in train_data:
            for label in inst.gold_labels:
                label_counter[label] += 1
        self._id2label = []

        for label, count in label_counter.most_common():
            self._id2label.append(label)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._label2id = reverse(self._id2label)
        if len(self._label2id) != len(self._id2label):
            print("serious bug: label dumplicated, please check!")

    def create_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword) - word_count
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if self._extword2id.get(values[0], self.UNK) != index:
                    print("Broken vocab or error embedding file, please check!")
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        return embeddings



    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.UNK) for x in xs]
        return self._label2id.get(xs, self.UNK)

    def id2label(self, xs):
        if isinstance(xs, list):
            return [self._id2label[x] for x in xs]
        return self._id2label[xs]


    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def char2id(self, xs):
        if isinstance(xs, list):
            return [self._char2id.get(x, self.UNK) for x in xs]
        return self._char2id.get(xs, self.UNK)

    def id2char(self, xs):
        if isinstance(xs, list):
            return [self._id2char[x] for x in xs]
        return self._id2char[xs]

    def bichar2id(self, xs):
        if isinstance(xs, list):
            return [self._bichar2id.get(x, self.UNK) for x in xs]
        return self._bichar2id.get(xs, self.UNK)

    def id2bichar(self, xs):
        if isinstance(xs, list):
            return [self._id2bichar[x] for x in xs]
        return self._id2bichar[xs]

    def char2id(self, xs):
        if isinstance(xs, list):
            return [self._char2id.get(x, self.UNK) for x in xs]
        return self._char2id.get(xs, self.UNK)

    def id2char(self, xs):
        if isinstance(xs, list):
            return [self._id2char[x] for x in xs]
        return self._id2char[xs]

    def bichar2id(self, xs):
        if isinstance(xs, list):
            return [self._bichar2id.get(x, self.UNK) for x in xs]
        return self._bichar2id.get(xs, self.UNK)

    def id2bichar(self, xs):
        if isinstance(xs, list):
            return [self._id2bichar[x] for x in xs]
        return self._id2bichar[xs]


    def extchar2id(self, xs):
        if isinstance(xs, list):
            return [self._extchar2id.get(x, self.UNK) for x in xs]
        return self._extchar2id.get(xs, self.UNK)

    def id2extchar(self, xs):
        if isinstance(xs, list):
            return [self._id2extchar[x] for x in xs]
        return self._id2extchar[xs]

    def extbichar2id(self, xs):
        if isinstance(xs, list):
            return [self._extbichar2id.get(x, self.UNK) for x in xs]
        return self._extbichar2id.get(xs, self.UNK)

    def id2extbichar(self, xs):
        if isinstance(xs, list):
            return [self._id2extbichar[x] for x in xs]
        return self._id2extbichar[xs]

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def label_size(self):
        return len(self._id2label)

    @property
    def char_size(self):
        return len(self._id2char)

    @property
    def bichar_size(self):
        return len(self._id2bichar)

    @property
    def extchar_size(self):
        return len(self._id2extchar)

    @property
    def extbichar_size(self):
        return len(self._id2extbichar)

def normalize_to_lowerwithdigit(str):
    str = str.lower()
    str = re.sub(r'\d', '0', str) ### replace digit 2 zero
    return str

def creatVocab(train_data, min_occur_count):
    word_counter = Counter()
    bichar_counter = Counter()
    char_counter = Counter()

    for inst in train_data:
        for w in inst.words:
            word_counter[w] += 1
        for c in inst.chars:
            char_counter[c] += 1
        for bc in inst.bichars:
            bichar_counter[bc] += 1

    return Vocab(word_counter, char_counter, bichar_counter, min_occur_count)


