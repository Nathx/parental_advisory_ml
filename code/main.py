from spark_model import SparkModel
import socket
from document import Document
from pyspark import SparkContext, SparkConf
from boto.s3.connection import S3Connection
from pyspark import SparkConf, SparkContext
import json
from datetime import datetime

def log_results(start_time, end_time, score, n_subs, clean_n_subs):
    with open('../logs/log.txt', 'a') as f:
       f.write('-'*40+'\n')
       duration = str(end_time - start_time).split('.')[0]
       f.write('Model: NaiveBayes\n')
       f.write('Number of subs: %s\n' % n_subs)
       f.write('Percentage subs parsed: %.1f%%\n' % (float(clean_n_subs) / n_subs))
       f.write('Time to run: %s\n' % duration)
       f.write('Accuracy: %.2f\n' % score)


if __name__ == '__main__':

    with open('/root/.aws/credentials.json') as f:
        CREDENTIALS = json.load(f)

    PATH_TO_DATA = '/vol0/en'
    # sc = SparkContext()
    APP_NAME = 'spark_model'
    conf = (SparkConf()
                .setAppName(APP_NAME)
                .set("spark.shuffle.service.enabled", True)
                .set("spark.dynamicAllocation.enabled", True)
                .set("spark.executor.cores", 2)
                .setMaster('spark://ec2-54-242-30-75.compute-1.amazonaws.com:7077'))
    sc = SparkContext(conf=conf, pyFiles=['document.py'])

    conn = S3Connection(CREDENTIALS['ACCESS_KEY'], CREDENTIALS['SECRET_ACCESS_KEY'])

    start_time = datetime.now()
    sm = SparkModel(sc, conn, PATH_TO_DATA, n_subs=100)
    sm.preprocess()
    subs, clean_subs = sm.n_subs, len(sm.labeled_paths)
    score = 0
#    sm.train()
#    score = sm.eval_score()
    end_time = datetime.now()
    log_results(start_time, end_time, score, subs, clean_subs)

    sc.stop()
