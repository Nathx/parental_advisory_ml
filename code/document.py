from boto.s3.key import Key
import socket
import xmltodict
from collections import OrderedDict
import gzip

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

    def get_sub(self):
        """Returns subtitle from file if it exists."""
        try:
            return self.contents['document']['s']
        except KeyError:
            print sub.keys()

    def load_file(self):
        if type(self.key) == Key:
            filename = 'file.xml'
            self.key.get_contents_to_filename('file.xml.gz')
            if self.key.name.endswith('.gz'):
                return gzip.GzipFile(fileobj=open(filename, 'rb'))
            else:
                return open(filename,'r')
        else:
            filename = self.key
            if filename.endswith('.gz'):
                return gzip.GzipFile(fileobj=open(filename, 'rb'))
            else:
                return open(filename,'r')


    def parse_xml(self):
        """
        Loads XML file and converts to OrderedDict
        """
        f = self.load_file()
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
            return [e[field] for e in elem]
        elif type(elem) == OrderedDict:
            return [elem[field]]

    def get_bag_of_words(self):
        """Returns list of all words."""
        return [word for id, t, sentence in self.contents for word in sentence]


    def parse_nb(self):
        """
        Parameters
        --------
        Returns RDD of LabeledPoint objects to be trained.
        """
        return (self.filename, LabeledPoint(self.label, self.vec))
