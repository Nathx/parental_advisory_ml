from spark_model import SparkModel
from document import Document
from datetime import datetime
import socket

from pyspark import SparkContext, SparkConf
from boto.s3.connection import S3Connection
from pyspark import SparkConf, SparkContext

import json
import sys
import click
import logging

def set_spark_context():
    APP_NAME = 'spark_model'
    conf = (SparkConf()
                .setAppName(APP_NAME)
                .set("spark.executor.cores", 4)
                .setMaster('spark://ec2-54-173-173-223.compute-1.amazonaws.com:7077'))
    sc = SparkContext(conf=conf, pyFiles=['document.py'])
    return sc

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', filename='../logs/log.txt',level=logging.DEBUG)
    logging.debug('-'*40)
    logging.debug('-'*40)
    logging.debug('Execution time: %s' % str(datetime.now()))

    with open('/root/.aws/credentials.json') as f:
        CREDENTIALS = json.load(f)

    # sc = SparkContext()
    sc = set_spark_context()

    conn = S3Connection(CREDENTIALS['ACCESS_KEY'], CREDENTIALS['SECRET_ACCESS_KEY'])

    model_type = 'naive_bayes'
    logging.debug('Model: %s' % model_type)

    sm = SparkModel(sc, conn, model_type=model_type)
    subs, clean_subs = sm.n_subs, len(sm.target)

    logging.debug('Files loaded.')
    logging.debug('Percentage subs parsed: %.1f%%' % (100*float(clean_subs) / subs))

    sm.train()

    logging.debug('Model trained.'')
    score = sm.eval_score()

    logging.debug('Accuracy: %.2f\n' % score)

    try:
        sm.RDD.saveAsPickleFile('stemmed_RDD.pkl')
        logging.debug('Saving complete.')
    except:
        logging.debug('Saving failed.')

    sc.stop()
