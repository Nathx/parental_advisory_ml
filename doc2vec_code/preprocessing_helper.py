# coding: utf-8

from pymongo import MongoClient
import pandas as pd
import numpy as np
import re

def import_comments():
    """
    Connect to Mongo and import comments.
    """
    client = MongoClient()
    db = client.movies
    coll = db.parentalguide

    return pd.DataFrame(
        list(
            coll.find({
        "$or": [{
            "alcohol_drugs_smoking": {
                "$ne": None
            }
        }, {
            "violence_gore": {
                "$ne": None
            }
        }, {
            "profanity": {
                "$ne": None
            }
        }, {
            "frightening_intense_scenes": {
                "$ne": None
            }
        }, {
            "sex_nudity": {
                "$ne": None
            }
        }]
    }, {
        "imdb_id": 1,
        "alcohol_drugs_smoking": 1,
        "violence_gore": 1,
        "sex_nudity": 1,
        "frightening_intense_scenes": 1,
        "profanity": 1,
        "_id": 0
    })))

def tokenize(comment):
    """
    Turn list of sentences into list of words.
    """
    if len(comment) == 0:
        return np.nan
    else:
        cp = re.compile('\w+')
        return cp.findall(' '.join(comment).lower())


def extract_grade(comment):
    """
    Pull out the grade from a comment if found.
    """
    comment = np.array(comment)
    candidates = np.where(comment == '10')[0]
    for i in candidates:
        if i == 0:
            if comment[1] == '10':
                return 10
        else:
            val = comment[i-1]
            try:
                val = int(val)
                if val < 11:
                    return val
            except:
                pass


def negative_single(row):
    """
    Look for negative singletons.
    """
    if row.n_words == 1:
        word = row.comment[0].lower()
        if word in ['none', 'nothing', 'nil', 'na', 'no', 'non', 'nope', 'nada']:
            return 'NEGATIVE'

    return

def prepare_train(comments_df):
    """
    Create features and labels for doc2vec.
    """
    comments_df['clean_comment'] = comments_df.comment.apply(lambda x: [w for w in x if re.match('[a-zA-Z]+', w)])
    comments_df['sentence_tag'] = comments_df['category'] + '_' + comments_df.index.astype(str)

    return comments_df


# In[4]:
def main():
    """
    Load comments and apply all required transformations.

    Returns:
    --
    Dataframe with tags ready for doc2vec training.
    """

    # Load comments from Mongo
    comments_df = import_comments()

    # Pivot to Series of all comments index by category and id
    comments_df = comments_df.set_index('imdb_id').unstack()

    # Transform list of sentences into list of words, remove punctuation
    comments_df = comments_df.apply(tokenize)

    # Move category from index to column
    comments_df = comments_df.dropna().reset_index().set_index('imdb_id')
    comments_df.columns = ['category', 'comment']

    comments_df['n_words'] = comments_df.comment.apply(len)
    comments_df['binary_target'] = comments_df.apply(negative_single, axis=1)
    comments_df['grade'] = comments_df.comment.apply(extract_grade)

    # Simplify labels
    comments_df.category = comments_df.category.replace([u'alcohol_drugs_smoking', u'violence_gore', u'sex_nudity',
           u'profanity', u'frightening_intense_scenes'], [u'DRUGS', u'VIOLENCE', u'NUDITY',
           u'PROFANITY', u'SCARY'])

    return comments_df
