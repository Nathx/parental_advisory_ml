import re

def craft_key(text):
    """
    Input: Human-readable text.
    Output: clean underscore_case key name with all words in input.
    """
    words = re.compile('\w+').findall(text)
    words = [w.lower().encode('ascii', 'ignore') for w in words]
    return '_'.join(words)
