import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

palette = [sns.color_palette('bright', 10)[i] for i in [7,8,1,5,2,6,0,3,4,9]]

labels = sorted(list(set(pd.read_csv('../data/training_transformer_full.csv')['Category'])))

def visualize(data):

    plt.figure(figsize=(12,8))
    sns.scatterplot(
        x='x0', y='x1',
        hue='y',
        hue_order=labels,
        palette=palette,
        data=data,
        alpha=1,
        s=20
    )

    #labels = list(range(10))

    #for i, p in enumerate(palette):
        #plt.scatter([], [], c=[p], label=labels[i])

    #plt.legend(list(range(1, 11)))
    plt.legend()
    plt.xlim([-80, 120])

    #plt.savefig('tsne_cluster.png')

    plt.show()

data = pd.read_csv('data/datasets/brand-mcat-desc-sep-stem-tfidf-tsvd50-tsne2-seed1.csv')

data['y'] = data['y'].map({i : labels[i] for i in range(10)})

visualize(data)
