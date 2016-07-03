from spark_model import SparkModel
from datetime import datetime
import multiprocessing as mp

from pyspark import SparkContext, SparkConf
from boto.s3.connection import S3Connection
from pyspark import SparkConf, SparkContext

import json
import sys
import click
import logging


def set_spark_context(local):

    APP_NAME = 'grid_search'
    n_cores = mp.cpu_count()
    conf = (SparkConf()
                .setAppName(APP_NAME)
                .set("spark.executor.cores", n_cores))

    if local:
        conf.setMaster('local[4]')
    else:
        with open('/root/spark-ec2/masters') as f:
            uri = f.readline().strip('\n')
        master = 'spark://%s:7077' % uri
        conf.setMaster(master)

    sc = SparkContext(conf=conf, pyFiles=['document.py'])

    return sc


def save_file(sm, rdd_path, lp_path):
    if rdd_path:
        sm.RDD.saveAsPickleFile(rdd_path)
    if lp_path and hasattr(sm, 'labeled_points'):
        sm.labeled_points.saveAsPickleFile(lp_path)
        



def build_model(sc, conn, **kwargs):
    return SparkModel(sc, conn, **kwargs)


def main(local, debug, save, train, **kwargs):
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
    if save:
        rdd_path = kwargs.pop('rdd_path')
        lp_path = kwargs.pop('lp_path')

    main_log.debug('Model: %s' % kwargs['model_type'])
    sm = build_model(sc, conn, debug=debug, **kwargs)

    main_log.debug('%s: Files loaded.' % str(datetime.now()))

    subs, clean_subs = sm.n_subs, sm.target.count()
    main_log.debug('Percentage subs parsed: %.1f%%' % (100*float(clean_subs) / subs))
    if train:
        sm.train()
        main_log.debug('%s: Model trained.' % str(datetime.now()))

        score = sm.eval_score()
        main_log.debug('Accuracy: %.2f\n' % score)
    if save:
        save_file(sm, rdd_path, lp_path)

    sc.stop()

@click.group()
def group():
    pass

@group.command()
@click.option('--n_subs', default=1)
@click.option('--model_type', default='naive_bayes')
@click.option('--test_size', default=.2)
@click.option('--debug', is_flag=True)
@click.option('--rdd_path', default=None)
@click.option('--lp_path', default=None)
@click.option('--save', is_flag=True)
@click.option('--local', is_flag=True)
@click.option('--train', is_flag=True)
def run(local, debug, save, train, **kwargs):
    main(local, debug, save, train, **kwargs)

if __name__ == '__main__':
    group()
