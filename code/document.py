from boto.s3.key import Key
import socket
import xmltodict
from collections import OrderedDict
from gensim.models.doc2vec import TaggedDocument
import re
import logging
import gzip

logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class Document(object):
    """
    This class represents a subtitle file.
    """

    def __init__(self, key, label):
        self.label = label
        self.key = key
        self.corrupted = False

        self.extract_content()

        self.compiler = re.compile('[\w-]+')

    def extract_content(self):
        parsed_xml = self.parse_xml()
        self.contents = self.extract_sub(parsed_xml)
        self.meta = self.extract_meta(parsed_xml)

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
        xml_dict = xmltodict.parse(data)

        return xml_dict

    def extract_row(self, row):
        """Returns informations attached to one row of a subtitle.
        """
        row_id, times, words = [], [], []
        if '@id' in row:
            row_id = row['@id']
        if 'time' in row:
            times = self.flatten_row(row['time'], '@value')
        if 'w' in row:
            words = self.flatten_row(row['w'], '#text')
        return row_id, times, words

    def extract_sub(self, parsed_xml):
        """
        Returns subtitle as a list of triplets (id, timestamps, words).
        """
        sentences = []
        if self.corrupted:
            return sentences
        else:
            doc = parsed_xml['document']

        if 's' in doc:
            sub_content = doc['s']
            if type(sub_content) == list:
                sentences = [self.extract_row(row) for row in sub_content]
            elif type(sub_content) == OrderedDict:
                sentences = [self.extract_row(sub_content)]
            else:
                raise TypeError('%s: Format not recognized.' % key.name)
        else:
            self.corrupted = True

        return sentences

    def extract_meta(self, parsed_xml):
        """
        Extract meta-data contained in subtitle file.
        """
        meta = {}
        if self.corrupted:
            return meta
        else:
            doc = parsed_xml['document']

        if 'meta' in doc:

            # grab movie's metadata
            if 'source' in doc['meta'] and \
                        type(doc['meta']['source']) == OrderedDict:

                meta.update(dict(doc['meta']['source']))

            # grab subtitle's metadata
            if 'conversion' in doc['meta'] and \
                        type(doc['meta']['conversion']) == OrderedDict:
                # extract keys of interest
                conversion = dict((k, v)
                        for k, v in doc['meta']['conversion'].iteritems()
                        if k in ['sentences', 'tokens', 'unknown_words', 'truecased_words'])

                meta.update(conversion)

        return meta

    def flatten_row(self, elem, field):
        """Flattens nested dictionaries in the XML file."""
        if type(elem) == list:
            return [e.get(field, '') for e in elem]
        elif type(elem) == OrderedDict:
            return [elem.get(field, '')]
        else:
            return []

    def clean(self, text):
        text = text.lower()
        words = self.compiler.findall(text)
        return [w for w in words if len(w) > 1]

    def get_bag_of_words(self):
        """Returns list of all words."""
        all_words = []
        for id, t, sentence in self.contents:
            all_words += [w for words in sentence
                                for w in self.clean(words)]
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
