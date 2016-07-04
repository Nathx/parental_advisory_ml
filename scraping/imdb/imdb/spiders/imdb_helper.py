import re
import pymongo
from scrapy.conf import settings

def craft_key(text):
    """
    Input: Human-readable text.
    Output: clean underscore_case key name with all words in input.
    """
    words = re.compile('\w+').findall(text)
    words = [w.lower().encode('ascii', 'ignore') for w in words]
    return '_'.join(words)

def fetch_ids():
    connection = pymongo.MongoClient(
        settings['MONGODB_SERVER'],
        settings['MONGODB_PORT']
    )
    db = connection[settings['MONGODB_DB']]
    coll = db[settings['MONGODB_COLLECTION']]
    return coll.distinct('imdb_id')
