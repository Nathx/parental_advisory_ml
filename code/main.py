from spark_model import SparkModel
from document import Document
from pyspark import SparkContext, SparkConf
from boto.s3.connection import S3Connection
from pyspark import SparkConf, SparkContext
import json

def main():

    with open('credentials') as f:
        CREDENTIALS = json.load(f)

    # sc = SparkContext()
    APP_NAME = 'spark_model'
    conf = SparkConf()\
            .setAppName(APP_NAME)\
            .set("IPYTHON", 1)\
            .setMaster('spark://ec2-54-242-30-75.compute-1.amazonaws.com:7077')

    sc = SparkContext(conf=conf)

    conn = S3Connection(CREDENTIALS['ACCESS_KEY'], CREDENTIALS['SECRET_ACCESS_KEY'])
    PATH_TO_DATA = 's3://subtitle-project/data/'

    sm = SparkModel(sc, conn, '1968')
    return sm.RDD.first()


if __name__ == '__main__':
    main()
