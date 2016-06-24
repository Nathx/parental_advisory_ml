from boto.s3.key import Key
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
            self.key.get_contents_to_filename('file.xml')
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
        # if self.key.name == 'subtitle_project,data/xml/en/1969/65063/59688.xml.gz':
        #     print f.read()

        # try:
        #     return xmltodict.parse(f.read())
        # except:
        #     return {}

        xml=''



        line = f.readline()

        while line:
            xml += line.strip()
            line = f.readline()

        try:
            xml_dict = xmltodict.parse(xml)
        except:
            print xml
            return

        f.close()

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

    def extract_sub(self):
        """
        Returns subtitle as a list of triplets (id, timestamps, words).
        """
        if 'document' in self.parsed_xml.keys():
            doc = self.parsed_xml['document']
        else:
            return []
        sentences = []

        if 's' in doc.keys():
            for row in doc['s']:
                sentences.append(self.extract_row(row))

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
