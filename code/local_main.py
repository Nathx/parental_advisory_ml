from spark_model import SparkModel
import socket
from document import Document
from pyspark import SparkContext, SparkConf
from boto.s3.connection import S3Connection
from pyspark import SparkConf, SparkContext
import json

if __name__ == '__main__':

    with open('../credentials.json') as f:
        CREDENTIALS = json.load(f)

    # sc = SparkContext()
    APP_NAME = 'spark_model'
    conf = (SparkConf()
                .setAppName(APP_NAME)
                .setMaster('local[4]'))
    print conf
    sc = SparkContext(conf=conf, pyFiles=['document.py'])
    PATH_TO_DATA = '../data/xml/en'

    conn = S3Connection(CREDENTIALS['ACCESS_KEY'], CREDENTIALS['SECRET_ACCESS_KEY'])

    sm = SparkModel(sc, conn, PATH_TO_DATA, n_subs=10)
    sm.preprocess()
    print sm.labeled_points.first()
    sm.train()
    print sm.eval_score()
