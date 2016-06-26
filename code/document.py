from boto.s3.key import Key
import socket
import xmltodict
from collections import OrderedDict
from gensim.models.doc2vec import TaggedDocument
import re
import logging
import gzip

logger.getLogger(_name_)
# logger.setLevel(logging.DEBUG)

class Document(object):
    """
    This class represents a subtitle file.
    """

    def __init__(self, key, label):
        self.label = label
        self.key = key
        self.corrupted = False

        self.parsed_xml = self.parse_xml()
        self.contents = self.extract_sub()

        self.contractions = re.compile(r"|-|\"")
        # all non alphanumeric
        self.symbols = re.compile(r'(\W+)', re.U)
        # single character removal
        self.singles = re.compile(r'(\s\S\s)', re.I|re.U)
        # separators (any whitespace)
        self.seps = re.compile(r'\s+')
        self.compiler = re.compile('\w+')

    def get_sub(self):
        """Returns subtitle from file if it exists."""
        try:
            return self.contents['document']['s']
        except KeyError:
            print sub.keys()

    def load_file(self):
        data = self.key.get_contents_as_string()
        return data


    def parse_xml(self):
        """
        Loads XML file and converts to OrderedDict
        """
        data = self.load_file()
        try:
            xml_dict = xmltodict.parse(data)
        except:
            self.corrupted = True
            return []

        return xml_dict

    def extract_row(self, row):
        """Returns informations attached to one row of a subtitle.
        """
        row_id, times, words = [], [], []
        if '@id' in row:
            try:
                row_id = row['@id']
            except:
                logging.info("Issue reading row: %s in file %s" % (row, self.key.name))
        if 'time' in row:
            times = self.flatten_row(row['time'], '@value')
        if 'w' in row:
            words = self.flatten_row(row['w'], '#text')
        return row_id, times, words

    def extract_sub(self):
        """
        Returns subtitle as a list of triplets (id, timestamps, words).
        """
        sentences = []
        if self.corrupted:
            return sentences
        else:
            doc = self.parsed_xml['document']

        if 's' in doc.keys():
            sub_content = doc['s']
            for row in sub_content:
                sentences.append(self.extract_row(row))
        else:
            self.corrupted = True

        return sentences

    def flatten_row(self, elem, field):
        """Flattens nested dictionaries in the XML file."""
        if type(elem) == list:
            return [e.get(field, '') for e in elem]
        elif type(elem) == OrderedDict:
            return [elem.get(field, '')]

    def clean(self, text):
        text = text.lower()
        text = self.contractions.sub('', text)
        text = self.symbols.sub(r' \1 ', text)
        text = self.singles.sub(' ', text)
        text = self.seps.sub(' ', text)
        return text

    def get_bag_of_words(self):
        """Returns list of all words."""
        all_words = []
        for id, t, sentence in self.contents:
            all_words += [self.clean(w) for words in sentence for w in self.compiler.findall(words)]
        return all_words

    def get_tagged_doc(self):
        """Returns tagged document as required by doc2vec."""
        return TaggedDocument(self.get_bag_of_words(), self.label)

    def parse_nb(self):
        """
        Parameters
        --------
        Returns RDD of LabeledPoint objects to be trained.
        """
        return (self.filename, LabeledPoint(self.label, self.vec))
