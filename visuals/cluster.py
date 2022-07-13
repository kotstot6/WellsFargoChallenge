
def title():

    print('++++++++++++++++++++')
    print('Script: cluster.py')
    print('Description: preprocess data via bag of words -> clustering')
    print('Author: Kyle Otstot')
    print('++++++++++++++++++++')
    print()

import sys
import argparse
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import hstack
import pickle

def get_args():

    settings = ' '.join(sys.argv[1:])

    print('===== Settings =====')
    print(settings)
    print('====================')
    print()

    parser = argparse.ArgumentParser(description='preprocess data via bag of words -> clustering')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--no_mcat', action='store_true', help='excludes merchant category from text feature')
    parser.add_argument('--no_desc', action='store_true', help='excludes description from text feature')
    parser.add_argument('--sep_atts', action='store_true', help='separates brands a/o merchant category a/o description')
    parser.add_argument('--no_stem', action='store_true', help='disable porter stemming')
    parser.add_argument('--no_tfidf', action='store_true', help='disable TF-IDF')
    parser.add_argument('--no_reduction', action='store_true', help='save the bag of words with full dimensionality')
    parser.add_argument('--tsvd_dim', type=int, default=50, help='dimension we reduce to via truncated SVD')
    parser.add_argument('--no_tsne', action='store_true', help='stop the dimensionality reduction after TSVD')
    parser.add_argument('--tsne_dim', type=int, default=2, help='dimension we reduce to via t-SNE')

    parser.set_defaults(exclude_mcat=False, exclude_desc=False, sep_atts=False, no_stem=False, no_tfdif=False, no_tsne=False)

    return parser.parse_args(), settings

def set_seed(seed):
    np.random.seed(seed)

def get_name(args):

    name = 'brand-' + ('' if args.no_mcat else 'mcat-') + ('' if args.no_desc else 'desc-')
    name += 'sep-' if args.sep_atts else ''
    name += ('' if args.no_stem else 'stem-') + ('' if args.no_tfdif else 'tfidf-')
    name += '' if args.no_reduction else 'tsvd' + str(args.tsvd_dim) + '-'
    name += '' if args.no_reduction or args.no_tsne else 'tsne' + str(args.tsne_dim) + '-'
    name += 'seed' + str(args.seed)

    return name

def bag_of_words(df, args):

    labels = sorted(list(set(df['Category'])))
    y = df['Category'].map({label : i for i, label in enumerate(labels)})

    text_atts = ['Brand 1', 'Merchant Category', 'Description']

    text = df['Message']

    for att in text_atts:
        text = text.apply(lambda x: ('###' + att).join(x.split(att)))

    no_atts = [False, args.no_mcat, args.no_desc]
    words = [text.apply(lambda x: x.split('###')[i+1]) for i, no_att in enumerate(no_atts) if not no_att]

    print('# of text attributes:', len(words))

    if not args.sep_atts:
        print('Combining text attributes...')
        for i, _ in enumerate(words[1:]):
            words[0] = words[0].combine(words[i+1], lambda x, y: x + ' ' + y)
        words = [words[0]]

        print('# of text attributes:', len(words))

    bags = []

    for i, _ in enumerate(words):

        print('Processing text attribute', i+1, '...')

        words[i] = words[i].map(lambda x: x.lower())
        words[i] = words[i].str.replace('[^\w\s]', '', regex=True)
        words[i] = words[i].apply(nltk.word_tokenize)

        if not args.no_stem:
            stemmer = PorterStemmer()
            words[i] = words[i].apply(lambda x: [stemmer.stem(y) for y in x])

        words[i] = words[i].apply(lambda x: ' '.join(x))


        bag = CountVectorizer().fit_transform(words[i])

        if not args.no_tfidf:
            bag = TfidfTransformer().fit_transform(bag)

        print('Bag', i+1, 'has shape', bag.shape)

        bags.append(bag)

    bag = hstack(tuple(bags))

    print('Final bag of words has shape', bag.shape)

    return bag, y

def bag_reduction(bag, args):

    X = TruncatedSVD(n_components=args.tsvd_dim).fit_transform(bag)
    print('Reduced via TSVD to shape', X.shape)

    if args.no_tsne:
        return X

    X = TSNE(n_components=args.tsne_dim, init='pca', verbose=2, random_state=args.seed).fit_transform(X)

    print('Reduced via TSNE to shape', X.shape)

    return X

def main(args, name):

    df = pd.read_csv('../data/training_final.csv')
    bag, y = bag_of_words(df, args)

    if args.no_reduction:

        data_dict = {'bag' : bag, 'y' : y}

        with open('data/datasets/' + name + '.pickle', 'wb') as f:
            pickle.dump(data_dict, f)

        return

    X = bag_reduction(bag, args)

    data = pd.DataFrame({ 'x' + str(i) : X[:,i] for i in range(X.shape[1])})
    data['y'] = y

    data.to_csv('data/datasets/' + name + '.csv', index=False)


if __name__ == '__main__':

    title()
    args, _ = get_args()
    set_seed(args.seed)
    name = get_name(args)
    main(args, name)
