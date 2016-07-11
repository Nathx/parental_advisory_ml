# This script experiments with word2vec to cluster the target based on MPAA
# descriptions.

from gensim.models import Word2Vec
import logging
import seaborn as sns
import pandas as pd
import numpy as np
import regex as re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA
from datetime import datetime
import string

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def tokenize(reason):
    reason = re.sub(ur'[^\P{P}-]+', ' ', reason.lower()).split()
    if 'for' in reason:
        reason = reason[reason.index('for')+1:]
    elif len(reason) < 3:
        reason = []
    else:
        reason = reason[2:]
    return [w for w in reason if not w in (stop + list(string.punctuation))]

def clean_data(coll):
    target_df = pd.DataFrame(list(coll.find()))

    clustering = target_df[['RATING', 'REASON']]
    clustering = clustering[clustering.REASON != ''].drop(938)
    return clustering

def build_w2b_mat(filename, vocab):
    model = Word2Vec.load_word2vec_format('GloVe-1.2/vectors.txt', binary=False)


    w2v_mat = np.zeros((len(model[vocab[0]]), len(vocab)))
    for j, word in enumerate(vocab):
        w2v_mat[:, j] = model[word]
    return w2v_mat

def build_dendrogram(w2v_mat):

    sim_vec = pdist(w2v_mat.T, 'cosine')
    sim_mat = squareform(sim_vec)

    link_mat = linkage(sim_mat, method='complete', metric='cosine')

    fig = plt.figure(figsize=(10,10))
    d = dendrogram(link_mat, truncate_mode='level', color_threshold=.09, orientation='left',
                  labels=vocab,
                   leaf_font_size=10)
    return d

def build_wordmap(w2v_mat):



    pca = PCA(n_components=2)
    pca.fit(w2v_mat.T)
    w2v_pca = pca.transform(w2v_mat.T)



    km = KMeans(n_clusters=6)
    labels = km.fit_predict(w2vt_mat.T)

    colors = 255 * ScalarMappable(cmap='Paired').to_rgba(np.unique(labels))[:, :3]
    hex_colors = ['#%02x%02x%02x' % (r, g, b) for r,g,b in colors]

    sns.set_style('dark')
    fig, ax = plt.subplots(1,1, figsize=(1.5,1.5))
    ax.axis('off')

    # ax = fig.add_subplot(111)

    for i in range(w2vt_pca.shape[0]):
        plt.text(w2vt_pca[i, 0], w2vt_pca[i, 1], str(vocab[i]),
                 fontdict={'color': hex_colors[labels[i]], 'size': 12})

    return ax

if __name__ == '__main__':
    # pull data
    client = MongoClient()
    db = client.movies
    coll = db.film_ratings

    clustering = clean_data(coll)
    print datetime.now(), "Data loaded."

    stop = stopwords.words('english')
    vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=.99, min_df=.01)
    tfidf = vectorizer.fit_transform(clustering.REASON.values)
    vocab = vectorizer.get_feature_names()
    print datetime.now(), "TFIDF built."

    filename = 'GloVe-1.2/vectors.txt'
    w2v_mat = build_w2b_mat(filename, vocab)

    fig, axes = plt.subplots(2,1, figsize=(10,15))
    axes[0] = build_dendrogram(w2v_mat)
    print datetime.now(), "Dendrogram generated."
    axes[1] = build_wordmap(w2v_mat)
    print datetime.now(), "PCA generated."
    fig.savefig('images/plot_' + filename)
    print datetime.now(), "File saved."
