from spark_model import SparkModel
import socket
from document import Document
from pyspark import SparkContext, SparkConf
from boto.s3.connection import S3Connection
from pyspark import SparkConf, SparkContext
import json
import sys
from datetime import datetime

def log_results(model_type, start_time, end_time, score, n_subs, clean_n_subs):
    with open('../logs/log.txt', 'a') as f:
       f.write('-'*40+'\n')
       duration = str(end_time - start_time).split('.')[0]
       f.write('Model: %s\n' % model_type)
       f.write('Number of subs: %s\n' % n_subs)
       f.write('Percentage subs parsed: %.1f%%\n' % (100*float(clean_n_subs) / n_subs))
       f.write('Time to run: %s\n' % duration)
       f.write('Accuracy: %.2f\n' % score)
    

if __name__ == '__main__':

    with open('/root/.aws/credentials.json') as f:
        CREDENTIALS = json.load(f)

    # sc = SparkContext()
    APP_NAME = 'spark_model'
    conf = (SparkConf()
                .setAppName(APP_NAME)
                .set("spark.executor.cores", 4)
                .setMaster('spark://ec2-54-173-173-223.compute-1.amazonaws.com:7077'))
    sc = SparkContext(conf=conf, pyFiles=['document.py'])

    conn = S3Connection(CREDENTIALS['ACCESS_KEY'], CREDENTIALS['SECRET_ACCESS_KEY'])
    
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'naive_bayes'

    start_time = datetime.now()
    sm = SparkModel(sc, conn, n_subs=1000, model_type=model_type)
    sm.preprocess()
    subs, clean_subs = sm.n_subs, len(sm.labeled_paths)
    sm.train()
    score = sm.eval_score()
    end_time = datetime.now()
    log_results(model_type, start_time, end_time, score, subs, clean_subs)

    sm = SparkModel(sc, conn, n_subs=10)
    print sm.tfidf.first()
    sc.stop()
