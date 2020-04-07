import inspect, sys, hashlib

# Hack around a warning message deep inside scikit learn, loaded by nltk :-(
#  Modelled on https://stackoverflow.com/a/25067818
import warnings
with warnings.catch_warnings(record=True) as w:
    save_filters=warnings.filters
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    import nltk
    warnings.filters=save_filters
try:
    nltk
except NameError:
    # didn't load, produce the warning
    import nltk
from nltk.corpus import brown
from nltk.tag import map_tag

assert map_tag('brown', 'universal', 'NR-TL') == 'NOUN', '''
Brown-to-Universal POS tag map is out of date.'''



