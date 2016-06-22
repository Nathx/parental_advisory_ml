
# coding: utf-8


import subtitle_featurizer as sf
import os
from pyspark.mllib.feature import HashingTF
import pyspark as ps
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
import pandas as pd
from pyspark.sql import SQLContext
import numpy as np
from pyspark.mllib.classification import NaiveBayes
from pyspark.ml.feature import IDF



def extract_labels(path):
    """
    Loads labeled dataframe and extract list of labeled subtitle ids.
    """
    labeled_df = pd.read_csv(path)
    labeled_df['IDSubtitle'] = labeled_df['IDSubtitle'].astype(int)
    return labeled_df.IDSubtitle.astype(str).values


def extract_filenames(labeled, directory):
    """
    Loops over the directory and grabs the filepath for each valid sub_id.
    """
    lab_filepaths = set()
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_id = name.split('.')[0]
            if (file_id in labeled):
                rating = labeled_df[labeled_df.IDSubtitle == int(file_id)].RATING.iloc[0]
                lab_filepaths.add((rating, os.path.join(root, name)))
    return lab_filepaths

def get_sub(xml_dict):
    """Returns subtitle from file if it exists."""
    try:
        return xml_dict['document']['s']
    except:
        print sub.keys()

def process_files(filepaths):
    """
    Transforms each subtitle file into its tokens.

    Parameters
    --------
    filepaths: list of strings
    List of all subtitle files to be processed as part of this corpus.

    Returns
    -----
    RDD of (key, value) pairs where:
    key: (label, filepath)
    value: list of tokens.
    """

     return sc.parallelize(list(filepaths)).map(lambda x: (x, sf.parse_xml(x[1]))).mapValues(get_sub)\
     .mapValues(sf.extract_sub).mapValues(lambda x: [word for id, t, sentence in x for word in sentence]).cache()

def extract_tfidf(rdd):
    """
    Converts each subtitle into its TFIDF representation.

    Parameters
    --------
    RDD of (key, value) pairs where:
    key: (label, filepath)
    value: list of tokens.

    Returns
    --------
    Spark DataFrame with columns:
    key: (label, filepath) tuple
    tf: Term-frequency Sparse Vector.
    IDF: TFIDF Sparse Vector.
    """
    htf = HashingTF(10000)
    tf = rdd.mapValues(htf.transform).cache()

    df = sqlContext.createDataFrame(tf.collect(), ['key', 'tf'])

    idf = IDF(inputCol='tf', outputCol='idf')
    idf_model = idf.fit(df)
    return idf_model.transform(df)

def parse_nb(doc):
    """
    Parameters
    --------
    doc: ((label, filepath), TFIDF)
    Returns RDD of LabeledPoint objects to be trained.
    """
    key, vec = doc
    key_dict = key.asDict()
    new_key = key_dict['_2']
    label = ratings.index(key_dict['_1'])
    return (new_key, LabeledPoint(label, vec))

def get_ratings(labeled_df):
    """Returns list of possible ratings."""
    return labeled_df.RATING.unique()


def train_and_evaluate(tfidf):
    """
    Trains a multinomal NaiveBayes classifier on TFIDF features.

    Parameters
    ---------
    Spark DataFrame with columns:
    key: (label, filepath) tuple
    tf: Term-frequency Sparse Vector.
    IDF: TFIDF Sparse Vector.

    Returns
    ---------
    model: MLLib NaiveBayesModel object, trained.
    test_score: Accuracy of the model on test dataset.
    """
    labeled_rdd = tfidf.select('key', 'idf').rdd.map(parse_nb).cache()
    train, test = labeled_rdd.randomSplit([.8, .2])
    nb = NaiveBayes()
    model = nb.train(train.values())
    test_data = test.values().cache()
    predictions = model.predict(test_data.map(lambda x: x.features)).collect()
    truth = test_data.map(lambda x: x.label).collect()
    test_score = (np.array(predictions) == np.array(truth)).mean()
    return model, test_score


if __name__ == '__main__':
    # creating Spark environment
    sc = ps.SparkContext('local[4]')
    sqlContext = SQLContext(sc)

    # fetching subtitles paths and ratings
    path = 'data/labeled_df.csv'
    sub_ids = extract_ids(path)
    directory = "/Users/nathankiner/galvanize/NLP_subs_project/OpenSubtitles2016/xml/en/"
    filenames = extract_filenames(sub_ids, directory)
    ratings = get_ratings(labeled_df)

    # loading and preprocessing files
    rdd = process_files(lab_filepaths)
    tfidf = extract_tfidf(rdd)

    # Training the model and outputting accuracy
    model, test_score = train_and_predict(tfidf)
    print "Test score:", test_score
