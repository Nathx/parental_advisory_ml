import numpy as np
import pandas as pd
import socket
from document import Document
import pyspark as ps
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.ml.feature import IDF


class SparkModel(object):

    def __init__(self, sc, conn, subset='', n_subs=0):
        self.context = sc
        self.conn = conn
        self.bucket = conn.get_bucket('subtitle_project')
        self.key_to_labels = self.bucket.get_key('data/labeled_df.csv')
        self.path_to_files = 'data/xml/en/' + subset

        self.labeled_paths = self.map_files()
        if n_subs:
          self.labeled_paths = self.labeled_paths[:n_subs]
        self.RDD = self.process_files()
        self.tfidf = self.extract_tfidf()

    def extract_labels(self):
        """
        Loads labeled dataframe and extract list of labeled subtitle ids.
        """
        labeled_df = pd.read_csv(self.key_to_labels)
        labeled_df['IDSubtitle'] = labeled_df['IDSubtitle'].astype(int)
        return labeled_df

    def map_files(self):
        """
        Loops over the directory and grabs the filepath for each valid sub_id.
        """
        labels = self.extract_labels()

        labeled_paths = {}

        sub_ids = labels.IDSubtitle.astype(str).values
        ratings = labels.RATING.values

        for key in self.bucket.list(prefix=self.path_to_files):

            filename = key.name.encode('utf-8').split('/')[-1]
            file_id = filename.split('.')[0]

            if (file_id in sub_ids):

                rating = ratings[np.where(sub_ids == file_id)][0]
                labeled_paths[key] = rating

        return labeled_paths.items()

    def unique_ratings(self):
        """Returns list of possible ratings."""
        return np.unique(self.ratings)

    def process_files(self):
        """
        Transforms each subtitle file into its tokens.

        Returns
        -----
        RDD of (key, value) pairs where:
        key: (label, filepath)
        value: list of tokens.
        """
        return self.context.parallelize(self.labeled_paths).map(lambda (key, label):
                                                        (key.name, Document(key, label))).cache()

    def extract_tfidf(self, num_features=10000, minDocFreq=1):
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
        htf = HashingTF(num_features)
        tf = self.RDD.mapValues(lambda x:
                                x.get_bag_of_words()).mapValues(
                                    htf.transform).cache()

        sqlContext = SQLContext(self.context)
        df = sqlContext.createDataFrame(tf.collect(), ['key', 'tf'])

        idf = IDF(inputCol='tf', outputCol='idf', minDocFreq=minDocFreq)
        idf_model = idf.fit(df)
        idf_rdd = idf_model.transform(df)

        return idf_rdd.select('key', 'idf').rdd.map(tuple).cache()
