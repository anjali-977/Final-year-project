import tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np


# building the classifier
def building_classifier():
   

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

   b_name = 'bert_en_uncased_L-12_H-768_A-12'
   tensorflowhub_handle_encoder = map_name_to_handle[b_name]
   tensorflowhub_handle_preprocess = map_model_to_preprocess[b_name]

   print(f'Name of choosen BERT model: {tensorflowhub_handle_encoder}')
   print(f'Equivalent Preprocessor: {tensorflowhub_handle_preprocess}')
#    bert_model = hub.load(tensorflowhub_handle_encoder)

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



built_classifier = building_classifier()

built_classifier.load_weights('./model/model_trained.ckpt').expect_partial() # model_trained.ckpt




# from sklearn.svm import SVC
# import pandas as pd

# def filterData(rawData=None):
#    if rawData is None:
#       df_test = pd.read_csv(r"./test.csv")
#    else:
#       rawData = rawData.replace(r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
#                       ' ')
#       rawData = rawData.replace(r'#', ' ')
#       rawData = rawData.replace(r'@', ' ')
#       f = lambda rawData: replace_emoticons(rawData)
#       f = lambda rawData: replace_emoji(rawData)
#       rawData = rawData.replace('[0-9]', '')
#       f = lambda l: " ".join(l.lower() for l in rawData.split())
#       rawData = rawData.replace('rt ', "")
#       f = lambda rawData: contractions.fix(rawData)
#       rawData = rawData.translate(str.maketrans('', '', string.punctuation))
#       rawData = rawData.strip()
#       rawData = rawData.replace('\s+', ' ')
#       rawData = rawData.replace('[^\w\s]', '')
#       stop_words = set(stopwords.words('english'))
#       f = lambda x: ' '.join(term for term in rawData.split() if term not in stop_words)
#       return rawData

      
# df = pd.read_csv(r"train.csv")
# df = df.drop_duplicates('text', keep='last')

# from sklearn.model_selection import train_test_split
# X_train, X_valid, y_train, y_valid = train_test_split(df['text'].tolist(), df['target'].tolist(),
#                                                       test_size=0.25, stratify=df['target'].tolist(),
#                                                       random_state=0)


# model_svm = SVC()
# model_svm.fit(X_train, y_train)

# # Prediction with class
# predictions_svm = model_svm.predict(X_valid)

# from sklearn.metrics import accuracy_score 
# acc_svm= accuracy_score(y_valid, predictions_svm)*100

# from sklearn.metrics import classification_report
# print(classification_report(y_valid, predictions_svm, target_names=['Fake','Real']))
