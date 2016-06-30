import numpy as np
import pandas as pd
import os

from document import Document

import pyspark as ps
from pyspark.mllib.feature import HashingTF, IDF, StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.mllib.classification import NaiveBayes, LogisticRegressionWithLBFGS

import nltk.data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import logging

pyspark_log = logging.getLogger('pyspark')
pyspark_log.setLevel(logging.INFO)


class SparkModel(object):

    def __init__(self, sc, conn, n_subs=0, test_size=.2, subset='',
                    model_type='naive_bayes', debug=False,
                    rdd_path=None, lp_path=None):
        # store parameters
        self.context = sc # SparkContext
        self.conn = conn # S3Connection
        self.n_subs = n_subs # Number of subs to process
        self.test_size = test_size # Fraction of the dataset to use as test
        self.model_type = model_type # naive_bayes, log_reg
        self.n_part = 100 # number of partitions
        self.rdd_path = rdd_path # path to processed data if provided
        self.lp_path = lp_path # Path to labeled points if provided
        self.debug = debug

        # Files location
        self.bucket = conn.get_bucket('subtitle_project')
        self.key_to_labels = self.bucket.get_key('data/labeled_df.csv')
        self.path_to_files = 'data/xml_unzipped/en/' + subset

        # Preprocess the data unless learning features are already provided
        if lp_path:
            self.labeled_points = self.context.pickleFile(lp_path)
            self.target = self.labeled_points.map(lambda (key, lp):
                                                (key, lp.label)).cache()
        else:
            self.preprocess(rdd_path)


    def preprocess(self, rdd_path):
        """
        Preprocessing pipeline.
        - Fetch target and matching filenames in S3
        - Pull files from S3 and preprocess
        - Stores RDD of (key, value) pairs with:
            key: Key object linking to the file in S3
            value: Stemmed Bag Of Words
        If RDD_path is provided, loads the data from there.


        Returns
        -----
        Self
        """
        if self.debug:
            self.target = self.context.parallelize([
                (self.bucket.get_key('data/xml_unzipped/en/1968/62909/6214054.xml'), 'G')
                ])

            self.RDD = self.process_files()
            self.target = self.labeled_points.map(lambda (key, lp): (key, lp.label))
        elif rdd_path:
            self.RDD = self.context.pickleFile(rdd_path)

            self.target = self.labeled_points.map(lambda (key, lp): (key, lp.label))

        else:
            self.target = self.map_files_to_target(self.n_subs)
            self.RDD = self.process_files()

        self.target.cache()

        if self.n_subs == 0:
            self.n_subs = self.target.count()

        return self


    def extract_labels(self, binary=True):
        """
        Loads labeled dataframe and extract list of labeled subtitle ids.
        """
        data = self.key_to_labels.get_contents_as_string()
        valid_ratings = self.context.broadcast([u'R', u'PG-13', u'PG', u'G', u'NC-17'])
        labels_rdd = self.context.parallelize(data.split('\n')) \
                        .filter(lambda line: line != '' and not 'IDSubtitle' in line) \
                        .map(lambda line: (line.split(',')[0], line.split(',')[1])) \
                        .filter(lambda (file_id, rating): rating in valid_ratings.value)
        if binary:
            labels_rdd = labels_rdd.map(lambda (file_id, rating): (file_id, (rating != 'R')*'NOT_' + 'R'))
        return labels_rdd.sortByKey().cache() # for lookups


    def map_files_to_target(self, n_subs=0, shuffle=False):
        """
        Loops over the directory and grabs the filepath for each valid sub_id,
        then finds the corresponding target in the CSV of labels.
        """
        labels_rdd = self.extract_labels()

        # now fully parallelized.
        target = self.context.parallelize(self.bucket.list(
            prefix=self.path_to_files)) \
            .map(lambda key: (key, key.name.encode('utf-8').split('/')[-1])) \
            .map(lambda (key, filename): (filename.split('.')[0], key)) \
            .join(labels_rdd).values()
        if self.n_subs:
            fraction = float(self.n_subs) / target.count()
            target = target.sample(withReplacement=False, fraction=fraction)

        return target

    def unique_ratings(self):
        """Returns list of possible ratings."""
        return self.target.values().distinct().collect()

    def process_files(self):
        """
        Transforms each subtitle file into its tokens.
        Stemmatization and stopwords removal are done here.

        Returns
        -----
        RDD of (key, value) pairs where:
        key: filepath
        value: Stemmed Bag Of Words.
        """
        rdd = self.target \
                .map(lambda (key, label): (key.name, Document(key, label)))


        clean_rdd = rdd.filter(lambda (key, doc): not doc.corrupted) \
                        .repartition(self.n_part) \
                        .cache()

        # remove corrupted files from list of paths
        self.target = clean_rdd.map(lambda (key, doc): (key, doc.label)).cache()

        # prepare for BOW cleaning
        nltk.data.path.append(os.environ['NLTK_DATA'])
        stop = self.context.broadcast(stopwords.words('english'))
        porter = PorterStemmer()

        # transform doc into stemmed bag of words
        clean_rdd = clean_rdd.mapValues(lambda x:
                                x.get_bag_of_words()).mapValues(
                                    lambda x: [porter.stem(word)
                                                for word in x
                                                if not word in stop.value])
        return clean_rdd

    def extract_features(self, feat='tfidf', **kwargs):
        """
        Converts each subtitle into its TF/TFIDF representation.
        Normalizes if necessary.

        Parameters
        --------
        Feat: 'tf' or 'tfidf'.
        kwargs: num_features, minDocFreq, or other arguments to be passed
        to the MLLib objects.

        Returns
        --------
        RDD of features with key.
        """

        # transform BOW into TF vectors
        num_features = kwargs.get('num_features', 10000)
        htf = HashingTF(num_features)
        feat_rdd = self.RDD.mapValues(htf.transform).cache()

        # transform TF vectors into IDF vectors
        if feat == 'tfidf':
            keys, tf_vecs = feat_rdd.keys(), feat_rdd.values()
            minDocFreq = kwargs.get('minDocFreq', 2)
            idf = IDF(minDocFreq=minDocFreq)
            idf_model = idf.fit(tf_vecs)
            idf_rdd = idf_model.transform(tf_vecs.map(lambda vec: vec.toArray()))
            feat_rdd = keys.zip(idf_rdd)

        if self.model_type == 'log_reg':
            normalizer = StandardScaler(withMean=True, withStd=True)
            keys, vecs = feat_rdd.keys(), feat_rdd.values()
            norm_model = normalizer.fit(vecs)
            norm_rdd = norm_model.transform(vecs.map(lambda vec: vec.toArray()))
            feat_rdd = keys.zip(norm_rdd)

        return feat_rdd

    def make_labeled_points(self, features):
        """
        Embed features and target into LabeledPoint object.
        """
        ratings = self.context.broadcast(self.unique_ratings())

        return features.join(self.target) \
                .map(lambda (k, (vec, label)): (k, LabeledPoint(ratings.value.index(label), vec))) \
                .cache()

    def predict(self, rdd):
        """
        Predict method for interfacing.
        """
        return self.model.predict(rdd)

    def eval_score(self):
        """
        Compute score on test dataset.
        """
        test_rdd = self.labeled_points.join(self.y_test) \
                        .repartition(self.n_part).cache()

        test_data = test_rdd.map(lambda (key, (lp, label)): lp.features)
        predictions = self.predict(test_data)

        truth = test_rdd.map(lambda (key, (lp, label)): label)


        self.score = truth.zip(predictions).map(lambda (y, y_pred): (y == y_pred)).mean()

        return self.score

    def make_train_test(self, test_size):
        self.y_train, self.y_test = self.target.randomSplit([1 - test_size,
                                                                    test_size])
        return self

    def train(self, feat='tfidf'):
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
        if not self.lp_path:
            self.labeled_points = self.make_labeled_points(self.extract_features())
        self.make_train_test(self.test_size)

        train_rdd = self.labeled_points.join(self.y_train) \
                        .map(lambda (key, (lp, label)): lp) \
                        .repartition(self.n_part).cache()

        if self.model_type == 'naive_bayes':
            nb = NaiveBayes()
            self.model = nb.train(train_rdd)

        elif self.model_type == 'log_reg':
            n_classes = len(self.unique_ratings())
            features = train_rdd.map(lambda lp: LabeledPoint(lp.label, lp.features.toArray()))
            logreg = LogisticRegressionWithLBFGS.train(features, numClasses=n_classes)
            self.model = logreg

        # elif self

        return self


    def train_doc2vec():
        pass
