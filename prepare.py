import sys
sys.path.append('/usr/local/lib/python3.5/dist-packages/')
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json
import warnings
warnings.filterwarnings("ignore")
import pickle
import sentencepiece as spm

module_url = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
embed = hub.Module(module_url)
tf.logging.set_verbosity(tf.logging.WARN)

module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))

with tf.Session() as sess:
    spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)
print("SentencePiece model loaded at {}.".format(spm_path))

def process_to_IDs_in_sparse_format(sp, sentences):
  # An utility method that processes sentences with the sentence piece processor
  # 'sp' and returns the results in tf.SparseTensor-similar format:
  # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape=(len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)

def embed_sentence_lite(sentences):
    messages = sentences
    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(
          encodings,
          feed_dict={input_placeholder.values: values,
                    input_placeholder.indices: indices,
                    input_placeholder.dense_shape: dense_shape})
    
    return message_embeddings

def find_closest(sentence_rep,query_rep,K):
    top_K = np.argsort(np.sqrt((np.sum(np.square(sentence_rep - query_rep),axis=1))))[:K]
    return top_K

def embed_sentences(sentences):
    message_embeddings = []
    user_embedding = []
    
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(sentences))

    return message_embeddings

def get_sentiments(json_body):
    headers = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': YOUR_KEY_HERE,
    }

    params = urllib.parse.urlencode({})

    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
    conn.request("POST", "/text/analytics/v2.0/sentiment?", json_body, headers)
    response = conn.getresponse()
    data = response.read()
    return data


f = open('res/your_sents.p','rb')
your_sentences = pickle.load(f)
f.close()

list_embeds = []
for i in range(0,len(your_sentences),500):
    print(i)
    list_embeds.append(embed_sentence_lite(your_sentences[i:i+500]))
    
your_embeddings = np.vstack(list_embeds)

f = open('res\\your_embeddings.p','wb')
pickle.dump(your_embeddings,f)
f.close()

f = open('res/other_sents.p','rb')
other_sentences = pickle.load(f)
f.close()

list_embeds = []
for i in range(0,len(other_sentences),500):
    print(i)
    list_embeds.append(embed_sentence_lite(other_sentences[i:i+500]))
    
other_embeddings = np.vstack(list_embeds)

import pickle
f = open('res/other_embeddings.p','wb')
pickle.dump(other_embeddings,f)
f.close()

with open('res/dilogues.p','rb') as F:
    you_to_other = pickle.load(F)

keys = list(you_to_other.keys())

list_embeds = []
for i in range(0,len(keys),500):
    print(i)
    list_embeds.append(embed_sentence_lite(keys[i:i+500]))
    
key_embeddings = np.vstack(list_embeds)

import pickle
f = open('res/key_embeddings.p','wb')
pickle.dump(key_embeddings,f)
f.close()