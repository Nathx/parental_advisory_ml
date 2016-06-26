import numpy as np
import pandas as pd
from document import Document
import multiprocessing as mp
from gensim.models.doc2vec import TaggedDocument

def create_doc(key, label):
    return Document(key, label)


class MPModel(object):

    def __init__(self, conn, subset='', n_subs=0, feat='tfidf', test_size=.2, shuffle=False):
        # store parameters
        self.conn = conn
        self.feat = feat
        self.n_subs = n_subs
        self.shuffle = shuffle
        self.test_size = test_size

        # create multiprocessing environment
        self.n_cpu = mp.cpu_count()
        self.pool = mp.Pool(processes=self.n_cpu)

        # find subtitle files
        self.bucket = conn.get_bucket('subtitle_project')
        self.key_to_labels = self.bucket.get_key('data/labeled_df.csv')
        self.path_to_files = 'data/xml/en/' + subset
        self.labeled_paths = self.map_files(self.n_subs)
        if n_subs == 0:
            self.n_subs = n_subs



    def preprocess(self):
        self.files = self.process_files()
        self.make_train_test(self.test_size)
        return self

    def extract_labels(self):
        """
        Loads labeled dataframe and extract list of labeled subtitle ids.
        """
        labeled_df = pd.read_csv(self.key_to_labels)
        labeled_df['IDSubtitle'] = labeled_df['IDSubtitle'].astype(int)
        return labeled_df

    def map_files(self, n_subs=0):
        """
        Loops over the directory and grabs the filepath for each valid sub_id.
        """
        labels = self.extract_labels()

        labeled_paths = []

        sub_ids = labels.IDSubtitle.astype(str).values
        ratings = labels.RATING.values

        for key in self.bucket.list(prefix=self.path_to_files):

            filename = key.name.encode('utf-8').split('/')[-1]
            file_id = filename.split('.')[0]

            if (file_id in sub_ids):

                rating = ratings[np.where(sub_ids == file_id)][0]
                labeled_paths.append((key, rating))

            if not self.shuffle and n_subs > 0 and len(labeled_paths) == n_subs:
                return labeled_paths

        if n_subs > 0:
            np.random.shuffle(labeled_paths)
            return labeled_paths[:n_subs]


        return labeled_paths

    def unique_ratings(self):
        """Returns list of possible ratings."""
        return np.unique(self.ratings)

    def process_files(self):
        """
        Transforms each subtitle file into its tokens.

        Returns
        -----
        List of Document instances representing each subtitle.
        """

        files = self.pool.map(lambda (key, label): create_doc(key, label), self.labeled_paths)
        clean_files = [doc for doc in files if not doc.corrupted]

        # update list of paths
        clean_paths = [doc.key for doc in clean_files]
        self.labeled_paths = [(key, label) for key, label in self.labeled_paths
                                            if key.name in doc]
        return clean_files

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
            self.files = self.process_files()

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

    def eval_score(self):
        paths = set(self.test_paths)
        test_rdd = self.labeled_points.filter(lambda (key, lp):
                                                key in paths)
        test_data = test_rdd.values().cache()
        predictions = self.model.predict(test_data.map(lambda x: x.features)).collect()
        truth = test_data.map(lambda x: x.label).collect()

        self.score = (np.array(predictions) == np.array(truth)).mean()

        return self.score

    def make_train_test(self, test_size):
        n_test = int(test_size*self.n_subs)
        filenames = [key.name for key, label in self.labeled_paths]
        np.random.shuffle(filenames)
        self.test_paths = filenames[:n_test]
        self.train_paths = filenames[n_test:]
        return self

    def train(self, feat='tfidf', model='nb'):
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

        self.labeled_points = self.get_labeled_points(self.extract_features())
        paths = set(self.train_paths)
        train_rdd = self.labeled_points.filter(lambda (key, lp):
                                                key in paths).values()

        nb = NaiveBayes()
        self.model = nb.train(train_rdd)

        return self

    def train_doc2vec(self, **kwargs):
        docs = [doc.get_tagged_doc() for doc in self.files]
        self.model = Doc2vec(docs, **kwargs)
        return self.model
