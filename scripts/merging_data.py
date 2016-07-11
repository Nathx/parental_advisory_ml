
# coding: utf-8

# This script maps together the different dataframes collected from IMDB and
# OpenSubtitles. It uses multiprocessing and Levenshtein distance to join
# movies on titles.

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
    """
    Find closest movie name in the next dataframe within a specific year. To
    be used with multiprocessing.

    -------
    Parameters:
    Args: df1 (pandas DataFrame),
        df2 (pandas DataFrame), col (string): column containing movie title.

    Returns:
    Df1 with new column containing closest movie title in df2 for this year.
    """
    df1, df2, col = args
    df1[col] = df1.TITLE.map(lambda x: closest_title(x, df2[col]))
    df1[col + '_distance'] = series_edit_distance(df1, 'TITLE', col)
    return df1


# In[69]:

def closest_title(title, series):
    """
    Return closest movie name in a given dataframe.

    -------
    Parameters:
    title (string): Movie name
    series: (pandas Series): Series to search for closest title
    """
    matches = difflib.get_close_matches(title, series)
    if matches:
        return matches[0]
    else:
        return ''


def series_edit_distance(df, col1, col2):
    """
    Compute Levenshtein distance between title column and closest title column.
    """

    def real_distance(row):
        return edit_distance(row[col1], row[col2]) / float(min(len(row[col1]), len(row[col2])) + 1)

    return df.apply(real_distance, axis=1)

def title_strip(title):
    """Lower case title and remove punctuation."""

    return re.sub(ur"\p{P}+", "", title.lower())

def get_closest_title(target_df, df):
    """Initiate processes to merge two dataframes year-by-year."""
    pool = mp.Pool(processes=4)
    col = 'year' if 'year' in df.columns else 'MovieYear'
    col2 = 'title' if 'title' in df.columns else 'MovieName'

    dfs = pool.map(map_year, [(target_df[target_df.YEAR == year], df[df[col] == year], col2)
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
