import numpy as np
import pandas as pd
from document import Document
import pyspark as ps
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.mllib.classification import NaiveBayes, LogisticRegressionWithLBFGS
from pyspark.ml.feature import IDF


class SparkModel(object):

    def __init__(self, sc, conn, subset='', n_subs=0, feat='tfidf', test_size=.2,
                    model_type='naive_bayes', debug=False):
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
        self.path_to_files = 'data/xml_unzipped/en/' + subset
        if debug:
            self.labeled_paths = [(self.bucket.get_key('data/xml_unzipped/en/1968/62909/6214054.xml'), 'G')]
        else:
            self.labeled_paths = self.map_files(self.n_subs)
        if n_subs == 0:
            self.n_subs = len(self.labeled_paths)



    def preprocess(self):
        self.RDD = self.process_files()
        self.labeled_points = self.get_labeled_points(self.extract_features())
        self.make_train_test(self.test_size)
        return self

    def extract_labels(self):
        """
        Loads labeled dataframe and extract list of labeled subtitle ids.
        """
        labeled_df = pd.read_csv(self.key_to_labels)
        labeled_df['IDSubtitle'] = labeled_df['IDSubtitle'].astype(int)
        return labeled_df

    def extract_id(self, key):
        filename = key.name.encode('utf-8').split('/')[-1]
        return filename.split('.')[0]

    def map_files(self, n_subs=0, shuffle=False):
        """
        Loops over the directory and grabs the filepath for each valid sub_id.
        """
        labels = self.extract_labels()

        labeled_paths = {}

        sub_ids = labels.IDSubtitle.astype(str).values
        ratings = labels.RATING.values
       
        if n_subs > -1:
            for key in self.bucket.list(prefix=self.path_to_files):

                file_id = self.extract_id(key)

                if (file_id in sub_ids):
                    rating = ratings[np.where(sub_ids == file_id)][0]
                    labeled_paths.append((key, rating))

                    if len(labeled_paths) == n_subs:
                        return labeled_paths
            return labeled_paths
        else:
            # same as above, but parallelized
            return self.context.parallelize(self.bucket.list(
                prefix=self.path_to_files)).map(lambda key:
                    (key, self.extract_id(key))).filter(lambda (key, file_id):
                        file_id in sub_ids).map(lambda (key, file_id):
                                (key, ratings[np.where(file_id == sub_ids)][0])).collect()

    def unique_ratings(self):
        """Returns list of possible ratings."""
        ratings = zip(*self.labeled_paths)[1]
        return list(np.unique(ratings))

    def process_files(self):
        """
        Transforms each subtitle file into its tokens.

        Returns
        -----
        RDD of (key, value) pairs where:
        key: (label, filepath)
        value: list of tokens.
        """
        rdd = self.context.parallelize(
                self.labeled_paths, 100).map(lambda (key, label):
                                        (key.name, Document(key, label)))

        clean_rdd = rdd.filter(lambda (key, doc): not doc.corrupted).cache()

        # update list of paths
        existing_paths = rdd.keys().collect()
        self.labeled_paths = [(key, label) for key, label in self.labeled_paths
                                            if key.name in existing_paths]
        return clean_rdd

    def extract_features(self, feat='tfidf', num_features=10000, minDocFreq=1):
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
        if not hasattr(self, 'RDD'):
            self.RDD = self.process_files()

        htf = HashingTF(num_features)
        tf = self.RDD.mapValues(lambda x:
                                x.get_bag_of_words()).mapValues(
                                    htf.transform).cache()

        sqlContext = SQLContext(self.context)
        df = sqlContext.createDataFrame(tf.collect(), ['key', 'tf'])

        idf = IDF(inputCol='tf', outputCol='idf', minDocFreq=minDocFreq)
        idf_model = idf.fit(df)
        idf_rdd = idf_model.transform(df).select('key', 'idf').rdd.map(tuple).cache()

        return idf_rdd

    def get_labeled_points(self, features):
        if self.feat == 'tfidf':
            ratings = self.unique_ratings()
            label_map = dict((k.name, ratings.index(v)) for k, v in self.labeled_paths)
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
        filenames = [key.name for key, label in self.labeled_paths]
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
