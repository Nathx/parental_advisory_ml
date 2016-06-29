from spark_model import SparkModel
from datetime import datetime
import socket

from pyspark import SparkContext, SparkConf
from boto.s3.connection import S3Connection
from pyspark import SparkConf, SparkContext

import json
import sys
import click
import logging


def set_spark_context(local):

    APP_NAME = 'spark_model'
    conf = (SparkConf()
                .setAppName(APP_NAME)
                .set("spark.executor.cores", 4))

    if local:
        conf.setMaster('local[4]')
    else:
        conf.setMaster('spark://ec2-54-173-173-223.compute-1.amazonaws.com:7077')

    sc = SparkContext(conf=conf, pyFiles=['document.py'])

    return sc


def save_file(filename, save_rdd):
    if filename:
        if save_rdd:
            sm.RDD.saveAsPickleFile(filename)
        else:
            sm.labeled_points.saveAsPickleFile(filename)



def build_model(sc, conn, **kwargs):
    return SparkModel(sc, conn, **kwargs)


def main(local, debug, save_rdd, **kwargs):
    logging.basicConfig(format='%(asctime)s %(message)s')
    main_log = logging.getLogger('main')
    main_log.setLevel(logging.DEBUG)
    handler = logging.FileHandler('../logs/log.txt')
    main_log.addHandler(handler)
    main_log.debug('-'*40)
    main_log.debug('-'*40)
    main_log.debug('Execution time: %s' % str(datetime.now()))

    # with open('~/.aws/credentials.json') as f:
    #     CREDENTIALS = json.load(f)

    sc = set_spark_context(local)
    conn = S3Connection()

    filename = kwargs.pop('filename')

    main_log.debug('Model: %s' % kwargs['model_type'])
    sm = build_model(sc, conn, debug=debug, **kwargs)

    main_log.debug('%s: Files loaded.' % str(datetime.now()))

    subs, clean_subs = sm.n_subs, sm.target.count()
    main_log.debug('Percentage subs parsed: %.1f%%' % (100*float(clean_subs) / subs))

    sm.train()
    main_log.debug('%s: Model trained.' % str(datetime.now()))

    score = sm.eval_score()
    main_log.debug('Accuracy: %.2f\n' % score)

    save_file(filename, save_rdd)

    sc.stop()

@click.group()
def group():
    pass

@group.command()
@click.option('--n_subs', default=1)
@click.option('--model_type', default='naive_bayes')
@click.option('--test_size', default=.2)
@click.option('--rdd_path')
@click.option('--lp_path')
@click.option('--debug', is_flag=True)
@click.option('--filename', default=None)
@click.option('--save_rdd', is_flag=True)
@click.option('--local', is_flag=True)
def run(local, debug, save_rdd, **kwargs):
    main(local, debug, save_rdd, **kwargs)

if __name__ == '__main__':
    group()
