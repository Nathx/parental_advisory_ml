from spark_model import SparkModel
from boto.s3.connection import S3Connection
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover, NGram, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier
import numpy as np


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
    conn = S3Connection()
    sqc = SQLContext(sc)
    sm = SparkModel(sc, conn, rdd_path='meta_rdd.pkl')

    bow_rdd = sm.RDD.map(lambda (key, (bow, meta)): (key, bow))
    bow_rdd = sm.RDD.join(sm.target).map(lambda (key, (bow, label)): (label, bow))

    remover = StopWordsRemover(inputCol="raw", outputCol="words")
    hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="word_counts",
                numFeatures=10000)
    tfidf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features",
                minDocFreq=20)
    indexer = StringIndexer(inputCol="string_label", outputCol="label")

    for model in [GBTClassifier(), RandomForestClassifier(), MultiLayerPerceptronClassifier()]:

        if type(model) == MultiLayerPerceptronClassifier:
            layers = [10000, 100, 2]
            model = MultiLayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128)

        pipeline = Pipeline(stages=[remover, hashingTF, tfidf, # scaler,
                                    indexer, model])
        scores = cross_val_score(pipeline, bow_rdd)
        print type(model), scores, scores.mean()
