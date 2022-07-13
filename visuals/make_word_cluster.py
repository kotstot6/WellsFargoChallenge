
def title():

    print('++++++++++++++++++++')
    print('Script: make_word_cluster.py')
    print('Description: use word clustering algorithm to output png or svg of the cluster')
    print('Author: Kyle Otstot')
    print('++++++++++++++++++++')
    print()

import sys
import argparse
import pickle
from wordcloud import WordCloud
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def get_args():

    settings = ' '.join(sys.argv[1:])

    print('===== Settings =====')
    print(settings)
    print('====================')
    print()

    parser = argparse.ArgumentParser(description='make word cluster with word frequencies and t-SNE')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cluster', type=str, default='brand-mcat-desc-stem-tfidf-tsvd50-tsne2-seed1.csv',
                                            help='cluster data for word positionings')
    parser.add_argument('--svg', action='store_true', help='output svg instead of png')
    parser.add_argument('--max_font_size', type=int, default=50, help='max font size for word cloud')
    parser.add_argument('--alpha', type=float, default=0.3, help='tuning parameter for word sizes')
    parser.add_argument('--eps1', type=float, default=30, help='neighborhood size')
    parser.add_argument('--eps2', type=float, default=1., help='proportion of neighborhood in consideration')
    parser.add_argument('--max_words', type=int, default=500, help='max number of words in word cloud')

    parser.set_defaults(svg=False)

    return parser.parse_args(), settings

def set_seed(seed):
    np.random.seed(args.seed)

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

    def image_pos(self):
        pos = (self.pos + 80) / 160
        return np.array([600 - 600 * pos[1], 900 * pos[0]])

def normalize_word_dict(word_dict, cluster):

    df_cluster = pd.read_csv('data/datasets/' + cluster)
    label_freqs = np.array(df_cluster['y'].value_counts().sort_index() / len(df_cluster))

    for label, words in word_dict.items():

        scores = np.array([word.score for word in words])
        norm_scores = label_freqs[label] * scores / scores.max()

        for i, score in enumerate(norm_scores):
            word_dict[label][i].score = score

    return word_dict

def color_func(word_dict):

    def get_color(word, font_size, position, orientation, random_state=None, **kwargs):

        palette = [sns.color_palette('bright', 10)[i] for i in [7,8,1,5,2,6,0,3,4,9]]

        def word_to_label(word):
            return [label for label, words in word_dict.items() if any([w.word == word for w in words])][0]

        r, g, b = tuple(round(255 * x) for x in list(palette[word_to_label(word)]))

        return 'rgb(' + str(r) + ',' + str(g) + ',' + str(b) + ')'

    return get_color

def get_word_freqs(word_dict):

    word_freqs = {}
    for words in word_dict.values():
        for word in words:
            word_freqs[word.word] = word.score

    return word_freqs

def get_pos_dict(word_dict):

    pos_dict = {}
    for words in word_dict.values():
        for word in words:
            pos_dict[word.word] = word.image_pos()

    return pos_dict

def main(args):

    with open('data/word_cloud_data.obj', 'rb') as f:
        word_dict = pickle.load(f)

    word_dict = normalize_word_dict(word_dict, args.cluster)
    word_freqs = get_word_freqs(word_dict)
    pos_dict = get_pos_dict(word_dict)

    wc = WordCloud(background_color='white', width=900, height=600,
                color_func=color_func(word_dict), pos_dict=pos_dict, prefer_horizontal=1,
                max_word_length=20, max_words=args.max_words,
                random_state=args.seed, max_font_size=args.max_font_size, alpha=args.alpha,
                eps1=args.eps1, eps2=args.eps2)

    wc.generate_from_frequencies(word_freqs)

    plt.figure(dpi=1200)
    plt.imshow(wc, interpolation='bilInear')
    plt.axis('off')

    ext = 'svg' if args.svg else 'png'

    file_name = ('data/wordcloud_figures/mfs-' + str(args.max_font_size) + '-alpha-' + str(args.alpha)
                + '-eps1-' + str(args.eps1) + '-eps2-' + str(args.eps2) + '-seed-' + str(args.seed) + '.' + ext)
    plt.savefig(file_name, format=ext, dpi=1200)


if __name__ == '__main__':

    title()
    args, _ = get_args()
    set_seed(args.seed)

    main(args)
