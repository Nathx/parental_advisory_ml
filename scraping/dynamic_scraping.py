from selenium import webdriver
from pyvirtualdisplay import Display
from pymongo import MongoClient
from itertools import product
import numpy as np
import dynamic_scraping_helper as h
from bs4 import BeautifulSoup

import os



def fetch_source(driver, url):
    with h.wait_for_page_load(driver):
        driver.get(url)
    with h.wait_for_page_load(driver):
        driver.find_element_by_xpath('//a[@href="javascript:handleFullResults();"]').click()

    driver.switch_to_window(driver.window_handles[1])
    try:
        source = driver.page_source
    except:
        with h.wait_for_page_load(driver):
            source = driver.page_source

    driver.close()
    driver.switch_to_window(driver.window_handles[0])

    return source

def html_to_doc(soup, y):
    divs = soup.body.findAll('div', attrs= {'style':'width:1000px;'})
    try:
        keys = [div.text for div in divs[0]]
    except:
        return ''
    docs = []

    for div in divs[1:]:
        doc = {}
        for k, v in zip(keys, div):
            doc[k] = v.text
        doc['YEAR'] = y
        docs.append(doc)
    return docs


if __name__ == '__main__':
    client = MongoClient()
    db = client.movies
    coll = db.film_ratings
    coll.remove({})

    url = "http://www.filmratings.com/search.html?filmYear=%s&filmRating=%s&x=35&y=11"

    years = np.arange(1968, 2017)
    ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'X', 'GP', 'M', 'M/PG']

    # path2phantom = os.getcwd() + '/phantomjs-2.1.1-macosx/bin/phantomjs'
    driver = webdriver.Firefox()
    # driver.set_window_size(800, 600)

    for y, r in product(years, ratings):
        crawl_url = url % (y, r)
        source = fetch_source(driver, crawl_url)

        rating_soup = BeautifulSoup(source, 'html.parser')
        documents = html_to_doc(rating_soup, y)
        if documents:
            coll.insert_many(documents)
        print "%s documents with rating %s in %s" % (len(documents), r, y)
        # time.sleep(3)
    finally:
        driver.quit()
