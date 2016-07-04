
from scrapy import Spider, Request
from bs4 import BeautifulSoup
import cPickle as pkl
import imdb_helper as h
import regex as re


class ImdbSpider(Spider):
    name = "imdb"
    allowed_domains = ["imdb.com"]

    def start_requests(self):
        # load full list of IMDB ids from OpenSubtitles
        with open('/Users/nathankiner/galvanize/NLP_subs_project/data/crawl_ids_full.pkl', 'r') as f:
            crawl_ids = pkl.load(f)
        # remove already crawled ids
        existing_ids = h.fetch_ids()
        crawl_ids = [id for id in crawl_ids if id not in existing_ids]
        url = 'http://www.imdb.com/title/tt%s/parentalguide'

        # send request
        for c_id in crawl_ids:
            request = Request(url % c_id, self.parse)
            request.meta['imdb_id'] = c_id
            yield request

    def parse(self, response):

        item = {}
        soup = BeautifulSoup(response.body, 'html.parser')
        item['title'] = soup.select('a.main')[0].text
        item['imdb_id'] = response.meta['imdb_id']

        # extract all country-specific ratings
        for elem in soup.select('.info'):
            key = h.craft_key(elem.h5.get_text())
            value = elem.div.get_text().lower()
            if key == 'mpaa':
                item[key + '_reason'] = value
            elif key == 'certification':
                countries = value.split(' / ')
                worldwide = {}
                for country in countries:
                    country_key, val = country.strip().split(':', 1)
                    worldwide[h.craft_key(country_key)] = val
                item[key] = worldwide
            else:
                item[key] = value

        # extract all comments section from parental guide
        # (violence, frightening scenes, nudity, etc.)
        ids = re.compile('swiki\.2\.\d$')
        for elem in soup.findAll(id=ids):
            key = h.craft_key(elem.h3.get_text())
            value = elem.find(attrs={'class':"display"})
            item[key] = list(value.stripped_strings)

        yield item
