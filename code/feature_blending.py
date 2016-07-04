from main import set_spark_context
from spark_model import SparkModel
from datetime import datetime
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd

from boto.s3.connection import S3Connection
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.ml.feature import StringIndexer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, GBTClassifier, NaiveBayes, RandomForestClassifier

import logging

logging.basicConfig(format='%(asctime)s %(message)s')
blend_log = logging.getLogger('blend_log')
blend_log.setLevel(logging.DEBUG)
handler = logging.FileHandler('../logs/blend.txt')
blend_log.addHandler(handler)


def create_pipeline(model_type, num_features=10000):
    """
    Defines pipeline from BOW to prediction.
    """

    remover = StopWordsRemover(inputCol="bow", outputCol="words")
    hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="word_counts", numFeatures=num_features)
    tfidf = IDF(inputCol=hashingTF.getOutputCol(),
                outputCol="features")

    if model_type == 'log_reg':
        model = LogisticRegression()
    elif model_type == 'gbt':
        model = GBTClassifier()
    elif model_type == 'naive_bayes':
        model = NaiveBayes()
    elif model_type == 'rf':
        model = RandomForestClassifier()

    return Pipeline(stages=[remover, hashingTF, tfidf,
                                model])


def load_target(key):
    """
    Loads precomputed doc2vec predictions for each category.
    """
    sentiment_df = pd.read_csv(key)
    sentiment_df['IDSubtitle'] = sentiment_df['IDSubtitle'].astype(int)
    return sentiment_df[['IDSubtitle', 'sentiment', 'prediction']]


def split_train_holdout(category_group, rdd):
    """
    For a given set of category labels, returns:
    Holdout: Subset of RDD with no label
    Train: Subset of RDD with label.
    """

    # convert pandas dataframe in RDD (subtitle_id, doc2vec prediction)
    target = sc.parallelize(category_group[['IDSubtitle', 'prediction']] \
                .to_dict(orient='records'), 150) \
                .map(lambda x: (str(int(x['IDSubtitle'])), float(x['prediction'])))

    # map id to key name
    ids = sm.RDD.map(lambda (key, bow): (key.split('/')[-1].split('.')[0], key))
    target = ids.join(target).map(lambda (file_id, (key, bow)): (key, bow))

    # build train and holdout (unlabeled) dataset
    holdout = sqc.createDataFrame(sm.RDD.subtractByKey(target), ['key', 'bow']).cache()
    train = target.join(sm.RDD).map(lambda (key, (pred, bow)): (key, pred, bow)).cache()

    return holdout, train

def get_predictions(pipeline, holdout, train_rdd, test_rdd):
    train_df = sqc.createDataFrame(train_rdd, ['key', 'label', 'bow']).persist()
    test_df = sqc.createDataFrame(test_rdd, ['key', 'label', 'bow']).persist()

    # train model
    model = pipeline.fit(train_df)

    # get probabilities for test and holdout set, then store in arrays
    test_pred = model.transform(test_df).rdd.map(lambda row: (row.key, float(row.probability[1])))
    holdout_pred = model.transform(holdout).rdd.map(lambda row: (row.key, row.probability[1]))

    return test_pred, holdout_pred

def make_blend(category, holdout, train, pipeline, num_folds):
    """
    - Trains the pipeline on (n-1) folds of the train dataset.
    - Predicts n-th fold.
    - Predicts holdout set.
    Repeat n times, then:
    - Concatenate predictions made on each fold to reconstruct prediction for
    the whole train set.
    - Average predictions (probabilities) on the holdout set to get a final prediction.
    - Concatenates both to reconstruct the feature for the whole RDD.
    """

    blend_log.debug('Begin blending category %s' % category)
    train = train.repartition(300)
    folds = train.randomSplit([1./num_folds]*num_folds)

    for i, test_rdd in enumerate(folds):
        blend_log.debug('Fold %s' % i)
        # exclude test fold, join all others and convert both to dataframes
        train_rdd = sc.union([fold for fold in folds if fold != test_rdd]).repartition(150)
        test_pred, holdout_pred = get_predictions(pipeline, holdout, train_rdd, test_rdd)

        if i == 0:
            blend_rdd = test_pred
            holdout_rdd = holdout_pred
        else:
            blend_rdd = blend_rdd.union(test_pred)
            holdout_rdd = holdout_rdd.join(holdout_pred)

    # average holdout predictions
    holdout_rdd = holdout_rdd.mapValues(lambda x: float(np.mean(x)))

    # create dataframes and concatenate
    holdout_df = sqc.createDataFrame(holdout_rdd, ['key', category])
    pred_df = sqc.createDataFrame(blend_rdd, ['key', category])

    blend_log.debug('Finish blending category %s' % category)

    return holdout_df.unionAll(pred_df)

def get_all_blends(sm, key, model_type, **kwargs):
    """
    Loop over all categories and reconstruct full dataset with one
    feature for each rating dimension.
    """

    target = load_target(key)
    pipeline = create_pipeline(model_type)
    features = []

    for category, group in target.groupby('sentiment'):
        holdout, train = split_train_holdout(group, sm.RDD)
        feature_vector = make_blend(category, holdout, train, pipeline, 5)
        features.append(feature_vector)

    features_df = features.pop()
    for feature in features:
        features_df = features_df.join(feature, on='key')

    return features_df



if __name__ == '__main__':


    blend_log.debug('-'*40)
    blend_log.debug('-'*40)
    blend_log.debug('Execution time: %s' % str(datetime.now()))

    sc = set_spark_context()
    sqc = SQLContext(sc)

    conn = S3Connection()
    bucket = conn.get_bucket('subtitle_project')
    key = bucket.get_key('data/mapped_subtitle_sentiment.csv')

    sm = SparkModel(sc, conn, rdd_path='rdd.pkl')
    features_df = get_all_blends(sm, key, 'log_reg')

    features_df.write.parquet('blended_features.parquet')
