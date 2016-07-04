from spark_model import SparkModel
from boto.s3.connection import S3Connection
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from datetime import datetime
import numpy as np
import logging


def set_spark_context():

    APP_NAME = 'grid_search'
    n_cores = 8
    conf = (SparkConf()
                .setAppName(APP_NAME)
                .set("spark.executor.cores", n_cores))

    with open('/root/spark-ec2/masters') as f:
        uri = f.readline().strip('\n')
    master = 'spark://%s:7077' % uri
    conf.setMaster(master)

    sc = SparkContext(conf=conf, pyFiles=['document.py'])

    return sc


def cross_val_score(pipeline, rdd, numFolds=3):
    cv_scores = []
    folds = rdd.randomSplit([1./numFolds]*numFolds)

    for test_rdd in folds:

        train_rdd = sc.union([fold for fold in folds if fold != test_rdd])

        train_df = sqc.createDataFrame(train_rdd, ['string_label', 'raw'])
        test_df = sqc.createDataFrame(test_rdd, ['string_label', 'raw'])

        model = pipeline.fit(train_df)
        test_output = model.transform(test_df)
        score = test_output.rdd.map(lambda row: row.label == row.prediction).mean()

        cv_scores.append(score)

    return cv_scores


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(message)s')
    nn_gridsearch = logging.getLogger('nn_gridsearch')
    nn_gridsearch.setLevel(logging.DEBUG)
    handler = logging.FileHandler('../logs/nn_gridsearch.txt')
    nn_gridsearch.addHandler(handler)
    nn_gridsearch.debug('-'*40)
    nn_gridsearch.debug('-'*40)
    nn_gridsearch.debug('Execution time: %s' % str(datetime.now()))

    # with open('~/.aws/credentials.json') as f:
    #     CREDENTIALS = json.load(f)

    sc = set_spark_context()

    conn = S3Connection()
    sqc = SQLContext(sc)
    sm = SparkModel(sc, conn, rdd_path='rdd.pkl')


    bow_rdd = sm.RDD.join(sm.target).map(lambda (key, (bow, label)): (label, bow)) \
            .sample(withReplacement=False, fraction=.5, seed=1)
    df = sqc.createDataFrame(bow_rdd, ['string_label', 'raw'])
    train_rdd, test_rdd = df.randomSplit([.8, .2], seed=1)
    results = []

    num_features = 5000
    min_doc_freq = 20
    layers = [[5000, 2056, 512, 128, 2], [5000, 1000, 128, 2], [5000, 100, 2], [5000, 5000, 2]]

    for l in layers:
        remover = StopWordsRemover(inputCol="raw", outputCol="words")
        hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="word_counts",
                              numFeatures=num_features)
        tfidf = IDF(inputCol=hashingTF.getOutputCol(),
                    outputCol="features", minDocFreq=min_doc_freq)
        indexer = StringIndexer(inputCol="string_label", outputCol="label")

        mlpc = MultilayerPerceptronClassifier(maxIter=100,
                                              layers=l,
                                              blockSize=128)

        pipeline = Pipeline(stages=[remover, hashingTF, tfidf,
                                    indexer, mlpc])

        model = pipeline.fit(train_rdd)
        df_output = model.transform(train_rdd)
        test_output = model.transform(test_rdd).select("label", "prediction")
        score = test_output.rdd.map(lambda row: row.label == row.prediction).mean()
        nn_gridsearch.debug("Layers: %s, Accuracy: %s" % (layers, score))
