from spark_model import SparkModel
import socket
from document import Document
from pyspark import SparkContext, SparkConf
from boto.s3.connection import S3Connection
from pyspark import SparkConf, SparkContext
import json

if __name__ == '__main__':

    with open('/root/.aws/credentials.json') as f:
        CREDENTIALS = json.load(f)

    # sc = SparkContext()
    APP_NAME = 'spark_model'
    conf = (SparkConf()
                .setAppName(APP_NAME)
                .setMaster('spark://ec2-54-242-30-75.compute-1.amazonaws.com:7077'))
    print conf
    sc = SparkContext(conf=conf, pyFiles=['document.py'])

    conn = S3Connection(CREDENTIALS['ACCESS_KEY'], CREDENTIALS['SECRET_ACCESS_KEY'])
    PATH_TO_DATA = 's3://subtitle-project/data/'

    sm = SparkModel(sc, conn, n_subs=10)
    print sm.tfidf.first()
    sc.stop()
