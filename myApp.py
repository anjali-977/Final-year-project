from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import re
from emoji import UNICODE_EMOJI
import contractions
import string
import sqlite3
from sqlite3 import Error
import os
import tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)




# building the classifier
def building_classifier():
   b_name = 'bert_en_uncased_L-12_H-768_A-12'

   map_name_to_handle = {
      'bert_en_uncased_L-12_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
      'bert_en_cased_L-12_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
      'bert_multi_cased_L-12_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
      'small_bert/bert_en_uncased_L-2_H-128_A-2':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-2_H-256_A-4':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-2_H-512_A-8':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-2_H-768_A-12':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-4_H-128_A-2':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-4_H-256_A-4':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-4_H-512_A-8':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-4_H-768_A-12':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-6_H-128_A-2':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-6_H-256_A-4':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-6_H-512_A-8':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-6_H-768_A-12':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-8_H-128_A-2':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-8_H-256_A-4':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-8_H-512_A-8':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-8_H-768_A-12':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-10_H-128_A-2':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-10_H-256_A-4':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-10_H-512_A-8':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-10_H-768_A-12':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
      'small_bert/bert_en_uncased_L-12_H-128_A-2':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
      'small_bert/bert_en_uncased_L-12_H-256_A-4':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
      'small_bert/bert_en_uncased_L-12_H-512_A-8':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
      'small_bert/bert_en_uncased_L-12_H-768_A-12':
         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
      'albert_en_base':
         'https://tfhub.dev/tensorflow/albert_en_base/2',
      'electra_small':
         'https://tfhub.dev/google/electra_small/2',
      'electra_base':
         'https://tfhub.dev/google/electra_base/2',
      'experts_pubmed':
         'https://tfhub.dev/google/experts/bert/pubmed/2',
      'experts_wiki_books':
         'https://tfhub.dev/google/experts/bert/wiki_books/2',
      'talking-heads_base':
         'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
   }

   # choosing the equivalent preprocessing technique
   # also in tensorflow_hub
   map_model_to_preprocess = {
      'bert_en_uncased_L-12_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'bert_en_cased_L-12_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/2',
      'small_bert/bert_en_uncased_L-2_H-128_A-2':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-2_H-256_A-4':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-2_H-512_A-8':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-2_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-4_H-128_A-2':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-4_H-256_A-4':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-4_H-512_A-8':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-4_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-6_H-128_A-2':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-6_H-256_A-4':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-6_H-512_A-8':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-6_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-8_H-128_A-2':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-8_H-256_A-4':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-8_H-512_A-8':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-8_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-10_H-128_A-2':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-10_H-256_A-4':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-10_H-512_A-8':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-10_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-12_H-128_A-2':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-12_H-256_A-4':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-12_H-512_A-8':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'small_bert/bert_en_uncased_L-12_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'bert_multi_cased_L-12_H-768_A-12':
         'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2',
      'albert_en_base':
         'https://tfhub.dev/tensorflow/albert_en_preprocess/2',
      'electra_small':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'electra_base':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'experts_pubmed':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'experts_wiki_books':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
      'talking-heads_base':
         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
   }

   tensorflowhub_handle_encoder = map_name_to_handle[b_name]
   tensorflowhub_handle_preprocess = map_model_to_preprocess[b_name]

   print(f'Name of choosen BERT model: {tensorflowhub_handle_encoder}')
   print(f'Equivalent Preprocessor: {tensorflowhub_handle_preprocess}')

   text = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
   # calling preprocessing layer i.e, creates tokens and replace each word by its ID from embedding table
   preprocessing_layer = hub.KerasLayer(tensorflowhub_handle_preprocess, name='preprocessing')
   encoder_text = preprocessing_layer(text)
   # calling bert model
   encoder = hub.KerasLayer(tensorflowhub_handle_encoder, trainable=True, name='BERT_encoder')
   outputs = encoder(encoder_text)
   # output for classification tokens and we get as 768 dimensions as specified
   net = outputs['pooled_output']
   # dropout on 768 dimensions
   net = tf.keras.layers.Dropout(0.1)(net)
   # final classification
   net = tf.keras.layers.Dense(1, activation="sigmoid", name='classifier')(net)
   return tf.keras.Model(text, net)


def check_news(filtered_data):
   built_classifier = building_classifier()

   # defining a loss function to check how accurate our prediction is
   # using binaryCrossEntropy since it is a binary classification
   predict_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
   binary_metrics = tf.metrics.BinaryAccuracy()
   # defining epochs
   # here Adam optimizer is used i.e., learning rate is decreased slowly for initial phase
   epochs = 4
   # df = pd.read_csv(r"train.csv")
   # df = df.drop_duplicates('text', keep='last')

   # from sklearn.model_selection import train_test_split
   # X_train, X_valid, y_train, y_valid = train_test_split(df['text'].tolist(), df['target'].tolist(),
   #                                                       test_size=0.25, stratify=df['target'].tolist(),
   #                                                       random_state=0)
   batch_size = 45
   seed = 97
   # df_train = tf.data.Dataset.from_tensor_slices((df['text'].tolist(), df['target'].tolist())).batch(batch_size)
   # df_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(batch_size)
   # print(df_train)

   # df_size = tf.data.experimental.cardinality(df_train).numpy()
   # total training steps
   # n_trainsteps = df_size * epochs
   # learning rate for initial phase (10 %)
   # n_initialsteps = int(0.1 * n_trainsteps)

   init_lr = 3e-5
   # optimizer_adamw = optimization.create_optimizer(init_lr=init_lr, num_train_steps=n_trainsteps,
   #                                                 num_warmup_steps=n_initialsteps, optimizer_type='adamw')

   # optimizer_adam = optimization.create_optimizer(init_lr=init_lr, optimizer_type='Adam')
   # optimizer_adam=tf.keras.optimizers.Adam(learning_rate=0.1,beta_1=0.9,beta_2=0.999,epsilon=1e-4,amsgrad=False,name='Adam')

   # built_classifier.compile(optimizer=optimizer_adamw, loss=predict_loss, metrics=binary_metrics)
   # fitting our built classifier on train dataset
   # print(f'Training the classifier with {tensorflowhub_handle_encoder}')
   # record = built_classifier.fit(x=df_train, validation_data=df_valid, epochs=epochs)
   built_classifier.load_weights('./model/model_trained.ckpt').expect_partial() # model_trained.ckpt
   # df_test_dl = pd.read_csv(r"test.csv")
   # predicting on test data
   predict_bert = built_classifier.predict([filtered_data])
   print(predict_bert)
   threshold = 0.8
   try:
      temp1 = predict_bert[0][0]
   except:
      temp1 = predict_bert

   if temp1 > threshold:
      result = 1
   else:
      result = 0

   return result


def replace_emoji(text):
   emoji_format = re.compile('['
                             u'\U0001F300-\U0001F5FF'
                             u'\U0001F600-\U0001F64F'
                             u'\U0001F680-\U0001F6FF'
                             u'\u2600-\u26FF\u2700-\u27BF'
                             u'\U0001F1E0-\U0001F1FF'
                             u'\U00002702-\U000027B0'
                             u'\U000024C2-\U0001F251'
                             ']+', flags=re.UNICODE)
   return emoji_format.sub(r'', text)

def auth_user(email=None, pwd=None):
   conn = None
   try:
      # if os.path.isfile(db_file):
      conn = sqlite3.connect(r"./MyDB")

   except Error as e:
      print(e)

   cursor = conn.cursor()

   if cursor.execute('SELECT PASSWORD FROM USERS where EMAIL=="'+str(email)+'"').fetchone()[0] == str(pwd):
      print('User Authenticated: ')
      conn.close()
      return True
   return False

def create_connection(db_file, table, uname=None, email=None, pwd=None, uid=None, news=None, result=None):
   """ create a database connection to a SQLite database """
   conn = None
   try:
      # if os.path.isfile(db_file):
      conn = sqlite3.connect(db_file)

      # Creating a cursor object using the cursor() method
      cursor = conn.cursor()
      if table == "USERS":
         # Creating table as per requirement
         sql = '''CREATE TABLE if not exists Users(
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            USER_NAME CHAR(20) NOT NULL,
            EMAIL CHAR(20) NOT NULL,
            PASSWORD CHAR(20) NOT NULL
         );'''
         cursor.execute(sql)
      elif table == "NEWS":
         # Creating table as per requirement
         sql = '''CREATE TABLE News(
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            USER_ID INTEGER NOT NULL,
            NEWS CHAR(300) NOT NULL,
            RESULTS CHAR(100) NOT NULL,
            FOREIGN KEY (USER_ID) REFERENCES USERS (ID)
         );'''
         cursor.execute(sql)
         # conn.close()

   except Error as e:
      print(e)

   finally:
      if conn:
         if table == "USERS":
            sql_insert = """INSERT OR IGNORE INTO Users (USER_NAME, EMAIL, PASSWORD) VALUES (?,?,?);"""
            data_record = (uname, email, pwd)
            cursor.execute(sql_insert, data_record)
            for user in cursor.execute('SELECT * FROM Users'):
               print('Inserted : ', user)

         elif table == "NEWS":
            sql_insert = """INSERT OR IGNORE INTO News (USER_ID, NEWS, RESULTS) VALUES (?,?,?);"""
            data_record = (uid, news, result)
            cursor.execute(sql_insert, data_record)
            for rec in cursor.execute('SELECT * FROM NEWS'):
               print('Inserted : ', rec)

         conn.commit()
         conn.close()

def get_uname(email, pwd):
   conn = None
   try:
      # if os.path.isfile(db_file):
      conn = sqlite3.connect(r"./MyDB")

   except Error as e:
      print(e)

   cursor = conn.cursor()

   if cursor.execute('SELECT PASSWORD FROM USERS where EMAIL=="' + str(email) + '"').fetchone()[0] == str(pwd):
      print('User Authenticated: ')
      uname = cursor.execute('SELECT USER_NAME FROM USERS where EMAIL=="' + str(email) + '"').fetchone()[0]
      conn.close()
      return uname

def get_uid(uname):
   conn = None
   try:
      # if os.path.isfile(db_file):
      conn = sqlite3.connect(r"./MyDB")

   except Error as e:
      print(e)

   cursor = conn.cursor()
   try:
      uid = cursor.execute('SELECT id FROM USERS where USER_NAME=="' + str(uname) + '"').fetchone()[0]
   except:
      uid = None
   conn.close()
   return uid


def replace_emoticons(text):
   emoticons = {
      u":‑\)": "smiley", u":\)": "smiley", u":-\]": "smiley",
      u":\]": "smiley", u":-3": "smiley", u":3": "smiley",
      u":->": "smiley", u":>": "smiley", u"8-\)": "smiley",
      u":o\)": "smiley", u":-\}": "smiley", u":\}": "smiley", u":-\)": "smiley",
      u":c\)": "smiley", u":\^\)": "smiley", u"=\]": "smiley",
      u"=\)": "smiley", u":‑D": "laugh", u":D": "laugh", u"8‑D": "laugh",
      u"8D": "laugh", u"X‑D": "laugh", u"XD": "laugh", u"=D": "laugh",
      u"=3": "laugh", u"B\^D": "laugh", u":\[": "upset", u":-\|\|": "upset",
      u":-\)\)": "happy", u":‑\[": "upset", u">:\[": "upset", u":\{": "upset",
      u"\(\^O\^\)": "Happy", u"\(\^o\^\)": "happy",
      u":‑\(": "upset", u":-\(": "upset", u":\(": "upset", u":‑c": "upset",
      u"\(o\.o\)": "Surprised", u"oO": "Surprised", u"\(\*￣m￣\)": "dissatisfied",
      u":c": "upset", u":‑<": "upset", u":<": "upset",

      u":@": "upset", u">:\(": "upset", u":'‑\(": "crying", u":'\(": "crying", u":'‑\)": "happy",
      u":'\)": "happy", u"D‑':": "Horror", u"D:<": "Disgust",
      u"D:": "Sad", u"D8": "dismay", u"D;": "dismay",
      u"D=": "dismay", u"DX": "dismay", u":‑O": "surprised",
      u":O": "surprised", u":‑o": "surprised", u":o": "surprised",
      u":-0": "shock", u"8‑0": "yawn", u">:O": "yawn",
      u":-\*": "affection", u":\*": "affection", u":X": "affection",
      u";‑\)": "wink", u";\)": "wink", u"\*-\)": "smirk", u"\*\)": "smirk", u";‑\]": "smirk",
      u";\]": "smirk", u";\^\)": "wink", u":‑,": "wink", u"=[(\\\)]": "skeptical",
      u";D": "wink", u":‑P": "Tongue sticking out", u":P": "Tongue sticking out",
      u"X‑P": "Tongue sticking out", u":-[.]": "hesitant", u"=/": "skeptical",
      u">:[(\\\)]": "hesitant", u">:/": "annoyed", u":[(\\\)]": "annoyed",
      u"XP": "Tongue sticking out", u":‑Þ": "Tongue sticking out", u":Þ": "Tongue sticking out,",
      u":b": "Tongue sticking out", u":‑/": "hesitant", u":/": "hesitant",
      u"d:": "Tongue sticking out", u"=p": "Tongue sticking out", u">:P": "Tongue sticking out",

      u":L": "skeptical", u"=L": "skeptical", u":S": "skeptical", u":‑\|": "Straight face",
      u":\|": "Straight face", u":$": "embarrassed", u":‑x": "tongue-tied", u":x": "tongue-tied",
      u":‑#": "tongue-tied", u":#": "tongue-tied", u":‑&": "tongue-tied", u":&": "tongue-tied",
      u"O:‑\)": "innocent", u"O:\)": "innocent", u"0:‑3": "innocent", u"0:3": "innocent",
      u"0:‑\)": "innocent", u"0:\)": "innocent", u":‑b": "tongue-tied", u"0;\^\)": "innocent",
      u">:‑\)": "evil", u">:\)": "evil", u"\}:‑\)": "evil", u"\}:\)": "evil", u"3:‑\)": "evil", u"3:\)": "evil",
      u">;\)": "evil", u"\|;‑\)": "cool", u"\|‑O": "bored", u":‑J": "Tongue-in-cheek",
      u"#‑\)": "Party all night", u"<:‑\|": "Dump", u"\(>_<\)": "Troubled", u"\(>_<\)>": "Troubled",
      u"%‑\)": "Drunk or confused", u"%\)": "Drunk or confused", u":-###..": "Being sick", u":###..": "Being sick",
      u"\(\＾ｖ\＾\)": "happy", u"\(\＾ｕ\＾\)": "Happy", u"\(\^\)o\(\^\)": "Happy",

      u'\(-"-\)': "Worried", u"\(ーー;\)": "Worried", u"\(\^0_0\^\)": "glasses",
      u"\(';'\)": "Baby", u"\(\^\^>``": "Nervous", u"\(\^_\^;\)": "Nervous",
      u"\(-_-;\)": "Nervous", u"\(~_~;\) \(・\.・;\)": "Nervous",
      u"\(-_-\)zzz": "Sleeping", u"\(\^_-\)": "Wink", u"\(\(\+_\+\)\)": "Confused", u"\(\+o\+\)": "Confused",
      u"\)\^o\^\(": "Happy", u":O o_O": "surprised", u"o_0": "surprised", u"o\.O": "amazed",
      u"\(o\|o\)": "Ultraman", u"\^_\^": "Joyful", u"\(\^_\^\)/": "Joyful",
      u"\(\^O\^\)／": "Joyful", u"\(\^o\^\)／": "Joyful", u"\(__\)": "respect", u"_\(\._\.\)_": "respect",
      u"<\(_ _\)>": "respect", u"<m\(__\)m>": "respect", u"m\(__\)m": "respect",
      u"m\(_ _\)m": "respect", u"\('_'\)": "Sad", u"\(/_;\)": "Sad", u"\(T_T\) \(;_;\)": "Sad", u"\(;_;": "Sad",
      u"\(;_:\)": "sad", u"\(;O;\)": "sad", u"\(:_;\)": "sad", u"\(ToT\)": "sad", u";_;": "sad",
      u";-;": "sad", u";n;": "sad", u";;": "sad", u"Q\.Q": "sad", u"T\.T": "sad", u"QQ": "Sad",
      u"Q_Q": "sad", u"\(-\.-\)": "Shame", u"\(-_-\)": "Shame", u"\(一一\)": "Shame", u"\(；一_一\)": "Shame",
      u"\(=_=\)": "Tired", u"\(=\^\·\^=\)": "cat",
      u"\(=\^\·\·\^=\)": "cat", u"=_\^=	": "cat", u"\(\.\.\)": "Looking down", u"\(\._\.\)": "Looking down",

      u"\^m\^": "Giggling with hand covering mouth", u"\(\・\・?": "Confusion",
      u"\(?_?\)": "Confusion", u">\^_\^<": "Normal Laugh", u"<\^!\^>": "Normal Laugh",
      u"\^/\^": "Normal Laugh", u"\（\*\^_\^\*）": "Normal Laugh", u"\(\^<\^\) \(\^\.\^\)": "Normal Laugh",
      u"\(^\^\)": "Normal Laugh",
      u"\(\^\.\^\)": "Normal Laugh", u"\(\^_\^\.\)": "Normal Laugh", u"\(\^_\^\)": "Normal Laugh",
      u"\(\^\^\)": "Normal Laugh",
      u"\(\^J\^\)": "Normal Laugh", u"\(\*\^\.\^\*\)": "Normal Laugh", u"\(\^—\^\）": "Normal Laugh",
      u"\(#\^\.\^#\)": "Normal Laugh",
      u"\（\^—\^\）": "Waving", u"\(;_;\)/~~~": "Waving", u"\(\^\.\^\)/~~~": "Waving",
      u"\(-_-\)/~~~ \($\·\·\)/~~~": "Waving", u"\(T_T\)/~~~": "Waving", u"\(ToT\)/~~~": "Waving",
      u"\(\*\^0\^\*\)": "Excited",
      u"\(\*_\*\)": "Amazed", u"\(\*_\*;": "Amazed", u"\(\+_\+\) \(@_@\)": "Amazed",
      u"\(\*\^\^\)v": "Laughing,Cheerful",
      u"\(\^_\^\)v": "Laughing,Cheerful", u"\(\(d[-_-]b\)\)": "Headphones,Listening to music",
      u"O_o": "surprised", u"\(‘A`\)": "snubbed"
   }
   emoticon_pattern = re.compile(u'(' + u'|'.join(i for i in emoticons) + u')')
   return emoticon_pattern.sub(r'', text)

def filterData(rawData=None):
   if rawData is None:
      df_test = pd.read_csv(r"./test.csv")
   else:
      rawData = rawData.replace(r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
                      ' ')
      rawData = rawData.replace(r'#', ' ')
      rawData = rawData.replace(r'@', ' ')
      f = lambda rawData: replace_emoticons(rawData)
      f = lambda rawData: replace_emoji(rawData)
      rawData = rawData.replace('[0-9]', '')
      f = lambda l: " ".join(l.lower() for l in rawData.split())
      rawData = rawData.replace('rt ', "")
      f = lambda rawData: contractions.fix(rawData)
      rawData = rawData.translate(str.maketrans('', '', string.punctuation))
      rawData = rawData.strip()
      rawData = rawData.replace('\s+', ' ')
      rawData = rawData.replace('[^\w\s]', '')
      stop_words = set(stopwords.words('english'))
      f = lambda x: ' '.join(term for term in rawData.split() if term not in stop_words)
      return rawData


@app.route('/index', methods=["GET", "POST"])
def index():
   if request.method == "GET":
      with open('dll.txt', 'r') as f:
         uname = f.readline()
      uid = get_uid(uname)
      if uid:
         return render_template("./index.html")
      return render_template("./login.html")


   elif request.method == "POST":
      return render_template("./index.html")

@app.route('/', methods=["GET", "POST"])
def login():
   if request.method == "GET":
      return render_template("./login.html")
   elif request.method == "POST":
      email = request.form.get("email")
      pwd = request.form.get("pwd")

      if auth_user(email, pwd):
         uname = get_uname(email, pwd)
         with open('dll.txt', 'w') as f:
            f.write(str(uname))
         return redirect(url_for("index"),code=307)  # render_template("./index.html")
      return render_template("/login.html")

@app.route('/register', methods=["GET", "POST"])
def register():
   if request.method == "GET":
      return render_template("./register.html")

   elif request.method == "POST":
      uname = request.form.get("uname")
      email = request.form.get("email")
      pwd = request.form.get("pwd")
      table= "USERS"
      create_connection(r"./MyDB", table, uname, email, pwd)
      return render_template("./login.html")  # render_template("./

@app.route('/predict', methods=["GET", "POST"])
def predict():
   if request.method == "GET":
      with open('dll.txt', 'r') as f:
         uname = f.readline()
      uid = get_uid(uname)
      if uid:
         return render_template("./predict.html", data="")
      return render_template("./login.html")

   elif request.method == "POST":
      news = request.form.get("news")
      table= "NEWS"
      with open('dll.txt', 'r') as f:
         uname = f.readline()

      uid = get_uid(uname)

      filtered_data = filterData(news)
      result = check_news(filtered_data)

      create_connection(r"./MyDB", table, uid=uid, news=news, result=result)
      if result:
         out = "Fake"
      else:
         out = "Real"
      return render_template("./predict.html", data=out)

   # return render_template("./predict.html")

@app.route('/about')
def about():
   return render_template("./about.html")


@app.route('/contact')
def contact():
   return render_template("./contact.html")

@app.route('/report')
def report():
   return render_template("./report.html")

if __name__ == '__main__':
   app.run(debug=False)

