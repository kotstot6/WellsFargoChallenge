
def title():

    print('++++++++++++++++++++')
    print('Script: word_cluster_data.py')
    print('Description: groups words found in transaction data together based on common categories')
    print('Author: Kyle Otstot')
    print('++++++++++++++++++++')
    print()

import sys
import argparse
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from inflection import singularize
import pickle

def get_args():

    settings = ' '.join(sys.argv[1:])

    print('===== Settings =====')
    print(settings)
    print('====================')
    print()

    parser = argparse.ArgumentParser(description='prepares words for clustering')

    parser.add_argument('--cluster', type=str, default='brand-mcat-desc-stem-tfidf-tsvd50-tsne2-seed1.csv',
                                                help='cluster data for word positionings')

    return parser.parse_args(), settings

class Word():

    def __init__(self, index, label, score, bag, vocab):

        self.index = index
        self.label = label
        self.score = score
        self.bag = bag
        self.vocab = vocab
        self.pos = self.get_pos()
        self.word = self.get_word()

    def get_pos(self):
        b1 = df_cluster['y'] == self.label
        b2 = (self.bag[:,self.index] > 0).toarray().reshape(-1)
        mask = np.logical_and(b1, b2)
        matches = df_cluster.values[mask, :2]
        pos =  np.median(matches, axis=0).reshape(-1)
        return pos

    def get_word(self):
        return [word for word, val in self.vocab.items() if int(val) == self.index][0]

def bag_of_words(words):

    words = words.map(lambda x: x.lower())
    words = words.str.replace('[^\w\s]', '', regex=True)
    words = words.apply(nltk.word_tokenize)
    words = words.apply(lambda xx: [singularize(x) for x in xx])
    words = words.apply(lambda x: ' '.join(x))

    counter = CountVectorizer(min_df=20)
    bag = counter.fit_transform(words)

    return bag, counter

def get_freqs(bag):

    labels = sorted(list(set(df_train['Category'])))

    freqs = np.zeros((len(labels), bag.shape[1]))

    for i, label in enumerate(labels):
        mask = df_train['Category'] == label
        freqs[i, :] = np.mean(bag[mask,:], axis=0)

    total = np.mean(bag, axis=0)
    freqs = (freqs - total)

    return freqs, labels

def get_word_dict(bag, freqs, labels, vocab):

    word_dict = {i : [] for i in range(len(labels))}

    for i in range(freqs.shape[1]):

        best_label = np.argmax(freqs[:,i])
        word_dict[best_label].append(Word(i, best_label, freqs[best_label, i], bag, vocab))

    return word_dict

def main():

    print('Creating bag of words...')
    words = df_train['Message']
    bag, counter = bag_of_words(words)

    print('Calculating relative frequencies...')
    freqs, labels = get_freqs(bag)

    print('Creating word dictionary...')
    word_dict = get_word_dict(bag, freqs, labels, vocab=counter.vocabulary_)

    print()
    print('Most frequent words')
    print('-------------------')

    for i, word_list in word_dict.items():
        print(labels[i], ':', end=' ')
        print(*[word.word for word in sorted(word_list, key=lambda x: -x.score)[:5]], sep=', ')

    with open('data/word_cloud_data.pickle', 'wb') as f:
        pickle.dump(word_dict, f)

if __name__ == '__main__':
    title()
    args, _ = get_args()

    # Get global data frames
    df_train = pd.read_csv('../data/training_final.csv')
    df_cluster = pd.read_csv('data/datasets/' + args.cluster)

    main()
