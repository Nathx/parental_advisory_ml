# coding: utf-8

import preprocessing_helper as h
from datetime import datetime
import numpy as np
import sys
import os

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument, LabeledSentence

import logging

# logging into .txt file
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger('__name__')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('../logs/doc2vec_log.txt')
logger.addHandler(handler)
for _ in xrange(3): logger.info("-"*50)
logger.info("Execution time - %s" % datetime.now())

def build_corpus(group):
    """
    Function to apply to a category subset. Returns a list of LabeledSentences to train doc2vec.
    """
    return group.apply(lambda row: LabeledSentence(row.clean_comment, [row.sentence_tag]), axis=1).values


def train_doc2vec(sentences, **kwargs):
    """
    Train doc2vec with best params from CV.
    Pass parameters in kwargs to overwrite.
    Returns model trained on sentences.
    """
    params = {
        'size':300,
        'window':8,
        'min_count':1,
        'sample':1e-4,
        'negative':5,
        'workers':4,
        'alpha':0.025,
        'min_alpha':0.025
    }

    # pop max_epochs and min_alpha from kwargs, throw rest in params
    max_epochs = kwargs.pop('max_epochs', 20)
    min_alpha = kwargs.pop('min_alpha', 0.002)
    params.update(kwargs)

    # setting up for manual alpha decay
    alpha = params['alpha']
    alpha_delta = (alpha - min_alpha) / max_epochs

    model = Doc2Vec(**params)
    model.build_vocab(sentences)

    for epoch in range(1, max_epochs+1):
        logger.info("- Starting epoch %s, alpha=%.3f" % (epoch, model.alpha))
        sentences = np.random.permutation(sentences)
        model.train(sentences)
        model.alpha -= alpha_delta  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    return model


if __name__ == '__main__':

    comments_df = h.main()
    comments_df = h.prepare_train(comments_df)

    models = {}

    for category, group in comments_df.groupby('category'):
        corpus = build_corpus(group)
        n = len(corpus)

        fname = '../data/' + category + '_doc2vec.model'

        if os.path.exists(fname):
            continue

        logger.info("- Begin training label %s with %s sentences" % (category, n))
        model = train_doc2vec(corpus, dm=0)
        logger.info("- End training label %s with %s sentences" % (category, n))

        # save models to file

        model.save(fname)
