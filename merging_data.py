
# coding: utf-8

# In[135]:

import pandas as pd
import numpy as np
from pymongo import MongoClient
from nltk.metrics.distance import edit_distance
from string import maketrans, punctuation
import multiprocessing as mp
import regex as re
import difflib
import os



def map_year(args):
    year, df1, df2, col = args
    df1[col] = df1.TITLE.map(lambda x: closest_title(x, df2[col]))
    df1[col + '_distance'] = series_edit_distance(df1, 'TITLE', col)
    return df1


# In[69]:

def closest_title(title, df):
    matches = difflib.get_close_matches(title, df)
    if matches:
        return matches[0]
    else:
        return ''


def series_edit_distance(df, col1, col2):


    def real_distance(row):
        return edit_distance(row[col1], row[col2]) / float(min(len(row[col1]), len(row[col2])) + 1)

    return df.apply(real_distance, axis=1)

def title_strip(title):
    return re.sub(ur"\p{P}+", "", title.lower())

def get_closest_title(target_df, df):
    pool = mp.Pool(processes=4)
    col = 'year' if 'year' in df.columns else 'MovieYear'
    col2 = 'title' if 'title' in df.columns else 'MovieName'

    dfs = pool.map(map_year, [(year, target_df[target_df.YEAR == year], df[df[col] == year], col2)
                              for year in target_df.YEAR.unique()])
    df = pd.concat(dfs)
    return df

if __name__ == '__main__':
    client = MongoClient()
    db = client.movies
    coll = db.film_ratings

    target_df = pd.DataFrame(list(coll.find()))
    target_df.TITLE = target_df.TITLE.apply(lambda x:
                                            title_strip(x[:-6].strip())
                                            )



    movie_df = pd.read_csv('data/movies_1968.csv')
    movie_df.title = movie_df.title.apply(lambda x: x.decode('utf-8', 'ignore'))
    movie_df.title = movie_df.title.apply(title_strip)
    os_df = pd.read_csv('data/subtitles_all_clean_1968.csv')
    os_df.MovieName = os_df.MovieName.apply(lambda x: x.decode('utf-8', 'ignore'))
    os_df.MovieName = os_df.MovieName.apply(title_strip)
    target_df = get_closest_title(target_df, os_df)
    target_df = get_closest_title(target_df, movie_df)

    merged_df = pd.merge(target_df, os_df, on='MovieName', how='left')
    merged_df = pd.merge(merged_df, movie_df, on='title', how='left')

    merged_df.to_csv('data/merged_df.csv', encoding='utf-8')
