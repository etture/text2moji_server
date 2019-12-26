import sys
from os.path import abspath, dirname, join
dir_name = dirname(dirname(abspath(__file__)))
sys.path.append(join(dir_name, 'DeepMoji'))

import json
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from deepmoji.global_variables import (
    PRETRAINED_PATH,
    NB_TOKENS,
    VOCAB_PATH
)
from deepmoji.model_def import (
    deepmoji_transfer,
    deepmoji_architecture,
    deepmoji_feature_encoding,
    deepmoji_emojis,
    get_weights_from_hdf5
)
from deepmoji.word_generator import WordGenerator
from deepmoji.create_vocab import VocabBuilder
from deepmoji.sentence_tokenizer import SentenceTokenizer, extend_vocab, coverage
from deepmoji.tokenizer import tokenize

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
emoji_list = pd.read_csv(join(dirname(dirname(__file__)), 'emoji_unicode.csv'), header=None, names=['unicode', 'tail'])
st = SentenceTokenizer(vocabulary, 50)
model = deepmoji_emojis(maxlen=50, weight_path=PRETRAINED_PATH)

global graph
graph = tf.get_default_graph()

@app.route('/getresults', methods=['GET', 'POST'])
@cross_origin(origin='*')
def get_results():
    if request.method == 'GET':
        return 'text2moji'
    elif request.method == 'POST':
        sentence = request.json.get('sentence', '')
        tokenized, _, _ = st.tokenize_sentences([sentence])
        with graph.as_default():
            prob = model.predict(tokenized)
        results = dict()
        for num, p in enumerate(top_elements(prob[0], 5)):
            order = num+1
            o_emoji_str = get_emoji_from_index(p).encode('unicode_escape')
            # print(o_emoji_str)
            # print(o_emoji_str[2])
            if o_emoji_str[:5] == '\U000':
                emoji_str = o_emoji_str[5:]
            else:
                emoji_str = o_emoji_str[2:]
            probability = round(prob[0][p], 5)
            if order == 1:
                results['first'] = {
                    'hexcode': emoji_str,
                    'probability': probability
                }
            elif order == 2:
                results['second'] = {
                    'hexcode': emoji_str,
                    'probability': probability
                }
            elif order == 3:
                results['third'] = {
                    'hexcode': emoji_str,
                    'probability': probability
                }
            elif order == 4:
                results['fourth'] = {
                    'hexcode': emoji_str,
                    'probability': probability
                }
            elif order == 5:
                results['fifth'] = {
                    'hexcode': emoji_str,
                    'probability': probability
                }
            print(o_emoji_str.encode('unicode_escape'), probability)

        new = {
            'first': {
                'hexcode': '1f62c',
                'probability': 0.342
            },
            'second': {
                'hexcode': '1f645',
                'probability': 0.239
            },
            'third': {
                'hexcode': '1f499',
                'probability': 0.151
            },
            'fourth': {
                'hexcode': '1f60a',
                'probability': 0.102
            },
            'fifth': {
                'hexcode': '2764',
                'probability': 0.087
            },
        }
        return jsonify(results)
        
def top_elements(array, k):
        ind = np.argpartition(array, -k)[-k:]
        return ind[np.argsort(array[ind])][::-1]

def get_emoji_from_index(idx):
    return emoji_list.iloc[idx]['unicode'].decode('unicode_escape')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8124)