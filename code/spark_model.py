import numpy as np
import pandas as pd
import os

from document import Document

import pyspark as ps
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.mllib.classification import NaiveBayes, LogisticRegressionWithLBFGS
from pyspark.ml.feature import IDF

import nltk.data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class SparkModel(object):

    def __init__(self, sc, conn, n_subs=0, test_size=.2,
                    model_type='naive_bayes', debug=False,
                    rdd_path=None, lp_path=None):
        # store parameters
        self.context = sc
        self.conn = conn
        self.feat = feat
        self.n_subs = n_subs
        self.test_size = test_size
        self.model_type = model_type

        # find subtitle files
        self.bucket = conn.get_bucket('subtitle_project')
        self.key_to_labels = self.bucket.get_key('data/labeled_df.csv')
        self.path_to_files = 'data/xml_unzipped/en/'

        # Preprocess the data unless learning features are already provided
        if lp_path:
            self.labeled_points = self.context.load(lp_path)
            self.target = self.labeled_points.map(lambda (key, lp):
                                                (key, lp.label)).collect()
        else:
            self.preprocess(rdd_path)



    def preprocess(self, rdd_path):
        """
        Preprocessing pipeline.
        - Fetch target and matching filenames in S3
        - Pull files from S3 and preprocess
        If RDD_path is provided, loads the data from there.
        """
        if debug:
            self.target = [(self.bucket.get_key('data/xml_unzipped/en/1968/62909/6214054.xml'), 'G')]
            self.RDD = self.process_files()
            # self.labeled_points = self.get_labeled_points(self.extract_features())
            # self.make_train_test(self.test_size)
            self.target = self.labeled_points.map(lambda (key, lp): (key, lp.label)).collect()
            # self.make_train_test(self.test_size)
        elif rdd_path:
            self.RDD = self.context.pickleFile(rdd_path)
            # self.labeled_points = self.get_labeled_points(self.extract_features())
            self.target = self.labeled_points.map(lambda (key, lp): (key, lp.label)).collect()
            # self.make_train_test(self.test_size)
        else:
            self.target = self.map_files_to_target(self.n_subs)
            self.RDD = self.process_files()
            # self.labeled_points = self.get_labeled_points(self.extract_features())
            # self.make_train_test(self.test_size)

        if self.n_subs == 0:
            self.n_subs = len(self.target)

        return self

    def


    def extract_labels(self):
        """
        Loads labeled dataframe and extract list of labeled subtitle ids.
        """
        labeled_df = pd.read_csv(self.key_to_labels)
        labeled_df['IDSubtitle'] = labeled_df['IDSubtitle'].astype(int)
        return labeled_df

    def extract_id(self, key):
        """
        Extract ID from filename.
        """
        filename = key.name.encode('utf-8').split('/')[-1]
        return filename.split('.')[0]

    def map_files_to_target(self, n_subs=0, shuffle=False):
        """
        Loops over the directory and grabs the filepath for each valid sub_id,
        then finds the corresponding
        """
        labels = self.extract_labels()

        target = []

        sub_ids = labels.IDSubtitle.astype(str).values
        ratings = labels.RATING.values
        if n_subs > -1:
            for key in self.bucket.list(prefix=self.path_to_files):

                file_id = self.extract_id(key)

                if (file_id in sub_ids):
                    rating = ratings[np.where(sub_ids == file_id)][0]
                    target.append((key, rating))

                    if len(target) == n_subs:
                        return target
            return target
        else:
            # same as above, but parallelized
            return self.context.parallelize(self.bucket.list(
                prefix=self.path_to_files)).map(lambda key:
                    (key, self.extract_id(key))).filter(lambda (key, file_id):
                        file_id in sub_ids).map(lambda (key, file_id):
                                (key, ratings[np.where(file_id == sub_ids)][0])).collect()

    def unique_ratings(self):
        """Returns list of possible ratings."""
        ratings = zip(*self.target)[1]
        return list(np.unique(ratings))

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
        rdd = self.context.parallelize(
                self.target, 100).map(lambda (key, label):
                                        (key.name, Document(key, label)))

        clean_rdd = rdd.filter(lambda (key, doc): not doc.corrupted).cache()

        # remove corrupted files from list of paths
        existing_paths = rdd.keys().collect()
        self.target = [(key, label) for key, label in self.target
                                            if key.name in existing_paths]

        # prepare for BOW cleaning
        nltk.data.path.append(os.environ['NLTK_DATA'])
        stop = stopwords.words('english')
        porter = PorterStemmer()

        # transform doc into stemmed bag of words
        clean_rdd = clean_rdd.mapValues(lambda x:
                                x.get_bag_of_words()).mapValues(
                                    lambda x: [porter.stem(word)
                                                for word in x
                                                if not word in stop])
        return clean_rdd

    def extract_features(self, feat='tfidf', **kwargs):
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

        # transform BOW into TF vectors
        num_features = kwargs.get('num_features', 10000)
        htf = HashingTF(num_features)
        tf = self.RDD.mapValues(htf.transform).cache()

        # transform TF vectors into IDF vectors
        sqlContext = SQLContext(self.context)
        df = sqlContext.createDataFrame(tf.collect(), ['key', 'tf'])

        minDocFreq = kwargs.get('minDocFreq', 2)
        idf = IDF(inputCol='tf', outputCol='idf', minDocFreq=minDocFreq)
        idf_model = idf.fit(df)
        idf_rdd = idf_model.transform(df).select('key', 'idf').rdd.map(tuple).cache()

        if self.model_type =
        normalizer = StandardScaler()

        return idf_rdd

    def get_labeled_points(self, features):
        if self.feat == 'tfidf':
            ratings = self.unique_ratings()
            label_map = dict((k.name, ratings.index(v)) for k, v in self.target)
            return features.map(lambda (k, v): (k, LabeledPoint(label_map.get(k), v))).cache()

    def predict(self, rdd):
        return self.model.predict(rdd)

    def eval_score(self):
        paths = set(self.test_paths)
        test_rdd = self.labeled_points.filter(lambda (key, lp):
                                                key in paths)
        test_data = test_rdd.values().cache()

        truth = test_data.map(lambda x: x.label).collect()
        predictions = self.predict(test_data.map(lambda lp: lp.features)).collect()

        self.score = (np.array(predictions) == np.array(truth)).mean()

        return self.score

    def make_train_test(self, test_size):
        n_test = int(test_size*self.n_subs)
        filenames = [key.name for key, label in self.target]
        np.random.shuffle(filenames)
        self.test_paths = filenames[:n_test]
        self.train_paths = filenames[n_test:]
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
        if not hasattr(self, 'train_paths'):
            self.preprocess()

        paths = set(self.train_paths)
        train_rdd = self.labeled_points.filter(lambda (key, lp):
                                                key in paths).values()

        if self.model_type == 'naive_bayes':
            nb = NaiveBayes()
            self.model = nb.train(train_rdd)

        elif self.model_type == 'log_reg':
            n_classes = len(self.unique_ratings())
            features = train_rdd.map(lambda lp: LabeledPoint(lp.label, lp.features.toArray()))
            logreg = LogisticRegressionWithLBFGS.train(features, numClasses=n_classes)
            self.model = logreg

        return self


    def train_doc2vec():
        pass
