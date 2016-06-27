from boto.s3.key import Key
import socket
import xmltodict
from collections import OrderedDict
from gensim.models.doc2vec import TaggedDocument
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
            if self.key.name.endswith('.gz'):
                filename = 'file.xml.gz'
                self.key.get_contents_to_filename('file.xml.gz')
                return gzip.GzipFile(fileobj=open(filename, 'rb'))
            else:
                xml_file = StringIO()
                self.key.get_contents_to_file(xml_file)
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
            return [e.get(field, '') for e in elem]
        elif type(elem) == OrderedDict:
            return [elem.get(field, '')]

    def get_bag_of_words(self):
        """Returns list of all words."""
        all_words = []
        COMPILER = re.compile('\w+')
        for id, t, sentence in self.contents:
            all_words += [w.lower() for w in COMPILER.findall(words) for words in sentence]
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
