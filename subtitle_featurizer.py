from collections import OrderedDict
import xmltodict
import gzip

def parse_xml(file_name):
    if file_name.endswith('.gz'):
        f = gzip.open(file_name, 'r')
    else:
        f=open(file_name,'r')
    xml=''

    while True:
        line=f.readline()
        if line=='':
            break
        xml+=line.strip()
    try:
        xml_dict = xmltodict.parse(xml)
    except:
        print xml
        return
    return xml_dict

def extract_row(row):
    row_id, times, words = [], [], []
    row_id = row['@id']
    if 'time' in row:
        times = flatten_row(row['time'], '@value')
    if 'w' in row:
        words = flatten_row(row['w'], '#text')
    return row_id, times, words

def extract_sub(doc):
    sentences = []
    for row in doc:
        sentences.append(extract_row(row))
    return sentences

def flatten_row(elem, field):
    if type(elem) == list:
        return [e[field] for e in elem]
    elif type(elem) == OrderedDict:
        return [elem[field]]
