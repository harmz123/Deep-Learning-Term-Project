# import libraries

import numpy as np
import keras.utils
import ssl
import nltk
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
import pickle as cPickle
from keras.layers import Input, MaxPooling1D
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPool1D
from keras.utils import to_categorical
from keras.callbacks import  EarlyStopping
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt



# Other neccessary dependency settings

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
pd.options.mode.chained_assignment = None

######                                                                                              #########
######               PRE-PROCESSING DATA                                                            #########
######                SOURCE: https://github.com/jshayiding/HybridCNN-fake-news-classifier          #########


# load dataset
train_file = pd.read_csv('train.tsv', sep='\t', header=None, encoding='utf-8')
test_file = pd.read_csv('test.tsv', sep='\t', header=None, encoding='utf-8')
va_file = pd.read_csv('valid.tsv', sep='\t', header=None, encoding='utf-8')

column_names = ['Id', 'Label', 'Statement', 'Subject', 'Speaker', 'Speaker Job', 'State Info', 'Party', 'BT', 'FC',
                'HT', 'MT', 'PF', 'Context']
train_file.columns, test_file.columns, va_file.columns = column_names, column_names, column_names

train_data = train_file[train_file.columns[~train_file.columns.isin(['Id', 'BT', 'FC', 'HT', 'MT', 'PF'])]]
test_data = test_file[test_file.columns[~test_file.columns.isin(['Id', 'BT', 'FC', 'HT', 'MT', 'PF'])]]
val_data = va_file[va_file.columns[~va_file.columns.isin(['Id', 'BT', 'FC', 'HT', 'MT', 'PF'])]]

train_data.head(3)

multi_labels_dict = {'false': 0, 'true': 1, 'pants-fire': 2, 'barely-true': 3, 'half-true': 4, 'mostly-true': 5}
binary_labels = {'false': 1, 'true': -1, 'pants-fire': 1, 'barely-true': 1, 'half-true': 0, 'mostly-true': -1}

train_data['multi_label'] = train_data['Label'].apply(lambda x: multi_labels_dict[x])
test_data['multi_label'] = test_data['Label'].apply(lambda x: multi_labels_dict[x])
val_data['multi_label'] = val_data['Label'].apply(lambda x: multi_labels_dict[x])

# def one_hot_label(label):
#     return to_categorical(multi_labels_dict[x], num_classes=6)

speakers = ['barack-obama', 'donald-trump', 'hillary-clinton', 'mitt-romney',
            'scott-walker', 'john-mccain', 'rick-perry', 'chain-email',
            'marco-rubio', 'rick-scott', 'ted-cruz', 'bernie-s', 'chris-christie',
            'facebook-posts', 'charlie-crist', 'newt-gingrich', 'jeb-bush',
            'joe-biden', 'blog-posting', 'paul-ryan']



#### Bug Fix by Uzoamaka Ezeakunne ###
speaker_dict = {'barack-obama':0, 'donald-trump':1, 'hillary-clinton':2, 'mitt-romney':3,
            'scott-walker':4, 'john-mccain':5, 'rick-perry':6, 'chain-email':7,
            'marco-rubio':8, 'rick-scott':9, 'ted-cruz':10, 'bernie-s':11, 'chris-christie':12,
            'facebook-posts':13, 'charlie-crist':14, 'newt-gingrich':15, 'jeb-bush':16,
            'joe-biden':17, 'blog-posting':18, 'paul-ryan':19}




def speaker_projection(speaker):
    if isinstance(speaker, str):
        speaker = speaker.lower()
        matched = [s for s in speakers if s in speaker]
        if len(matched) > 0:
            return speaker_dict[matched[0]]
        else:
            return 19
    #### Bug Fix by Uzoamaka Ezeakunne ###
    else:
        return 19


##
train_data['speaker_id'] = train_data['Speaker'].apply(speaker_projection)
test_data['speaker_id'] = test_data['Speaker'].apply(speaker_projection)
val_data['speaker_id'] = val_data['Speaker'].apply(speaker_projection)

## Map job
job_list = ['president', 'u.s. senator', 'governor', 'president-elect', 'presidential candidate',
            'u.s. representative', 'state senator', 'attorney', 'state representative', 'congress', 'others']

job_dict = {'president': 0, 'u.s. senator': 1, 'governor': 2, 'president-elect': 3, 'presidential candidate': 4,
            'u.s. representative': 5, 'state senator': 6, 'attorney': 7, 'state representative': 8, 'congress': 9,
            'others': 10}


## Map job

def job_projection(job):
    if isinstance(job, str):
        job = job.lower()
        matched_job = [j for j in job_list if j in job]
        if len(matched_job) > 0:
            return job_dict[matched_job[0]]
        else:
            return 10
    else:
        return 10


## job projection output

train_data['job_id'] = train_data['Speaker Job'].apply(job_projection)
test_data['job_id'] = test_data['Speaker Job'].apply(job_projection)
val_data['job_id'] = val_data['Speaker Job'].apply(job_projection)

### Map political parties
party_dict = {'republican': 0, 'democrat': 1, 'none': 2, 'organization': 3, 'newsmaker': 4, 'rest': 5}


def map_political_party(party):
    if party in party_dict:
        return party_dict[party]
    else:
        return 5


##
train_data['party_id'] = train_data['Party'].apply(map_political_party)
test_data['party_id'] = test_data['Party'].apply(map_political_party)
val_data['party_id'] = val_data['Party'].apply(map_political_party)

## Map states
all_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
              'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
              'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
              'Maine' 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
              'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
              'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
              'North Carolina', 'North Dakota', 'Ohio',
              'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
              'South  Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah',
              'Vermont', 'Virginia', 'Washington', 'West Virginia',
              'Wisconsin', 'Wyoming']

states_dict = {'wyoming': 48, 'colorado': 5, 'washington': 45, 'hawaii': 10,
               'tennessee': 40, 'wisconsin': 47, 'nevada': 26, 'north dakota': 32,
               'mississippi': 22, 'south dakota': 39, 'new jersey': 28, 'oklahoma': 34,
               'delaware': 7, 'minnesota': 21, 'north carolina': 31, 'illinois': 12,
               'new york': 30, 'arkansas': 3, 'west virginia': 46, 'indiana': 13,
               'louisiana': 17, 'idaho': 11, 'south  carolina': 38, 'arizona': 2,
               'iowa': 14, 'mainemaryland': 18, 'michigan': 20, 'kansas': 15,
               'utah': 42, 'virginia': 44, 'oregon': 35, 'connecticut': 6, 'montana': 24,
               'california': 4, 'massachusetts': 19, 'rhode island': 37, 'vermont': 43,
               'georgia': 9, 'pennsylvania': 36, 'florida': 8, 'alaska': 1, 'kentucky': 16,
               'nebraska': 25, 'new hampshire': 27, 'texas': 41, 'missouri': 23, 'ohio': 33,
               'alabama': 0, 'new mexico': 29, 'rest': 50}


def state_projection(state):
    if isinstance(state, str):
        state = state.lower()
        if state in states_dict:
            return states_dict[state]
        else:
            if 'washington' in state:
                return states_dict['washington']
            else:
                return 50
    else:
        return 50


## state mapping output:
train_data['state_id'] = train_data['State Info'].apply(state_projection)
test_data['state_id'] = test_data['State Info'].apply(state_projection)
val_data['state_id'] = val_data['State Info'].apply(state_projection)

## map subject
subject_list = ['health', 'tax', 'immigration', 'election', 'education',
                'candidates-biography', 'economy', 'gun', 'jobs', 'federal-budget', 'energy', 'abortion',
                'foreign-policy']

subject_dict = {'health': 0, 'tax': 1, 'immigration': 2, 'election': 3, 'education': 4,
                'candidates-biography': 5, 'economy': 6, 'gun': 7, 'jobs': 8, 'federal-budget': 9,
                'energy': 10, 'abortion': 11, 'foreign-policy': 12, 'others': 13}


## mapping subject
def subject_projection(subject):
    if isinstance(subject, str):
        subject = subject.lower()
        matched_subject = [subj for subj in subject_list if subj in subject]

        if len(matched_subject) > 0:
            return subject_dict[matched_subject[0]]
        else:
            return 13
    else:
        return 13


##
train_data['subject_id'] = train_data['Subject'].apply(subject_projection)
test_data['subject_id'] = test_data['Subject'].apply(subject_projection)
val_data['subject_id'] = val_data['Subject'].apply(subject_projection)

## Context mapping
Context_list = ['news release', 'interview', 'tv', 'radio',
                'campaign', 'news conference', 'press conference', 'press release',
                'tweet', 'facebook', 'email']

Context_dict = {'news release': 0, 'interview': 1, 'tv': 2, 'radio': 3,
                'campaign': 4, 'news conference': 5, 'press conference': 6, 'press release': 7,
                'tweet': 8, 'facebook': 9, 'email': 10, 'others': 11}


def Context_projection(context):
    if isinstance(context, str):
        context = context.lower()
        matched_context = [cntx for cntx in Context_list if cntx in context]
        if len(matched_context) > 0:
            return Context_dict[matched_context[0]]
        else:
            return 11
    else:
        return 11


## context projection output
train_data['context_id'] = train_data['Context'].apply(Context_projection)
test_data['context_id'] = test_data['Context'].apply(Context_projection)
val_data['context_id'] = val_data['Context'].apply(Context_projection)

### tokenize fake news statement and build vocabulary
vocab_dict = {}

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(train_data['Statement'])
vocab_dict = tokenizer.word_index
cPickle.dump(tokenizer.word_index, open("vocab.p", "wb"))
print("vocab dictionary is created")
print("saved vocan dictionary to pickle file")

vocab_length = len(vocab_dict.keys())
num_steps = 32

# Meta data related hyper params
num_party = 6
num_state = 51
num_context = 12
num_job = 11
num_sub = 14
num_speaker = 20
embedding_dims = 300
max_features = len(tokenizer.word_index) + 1

vocab_length = len(vocab_dict.keys())


### create embedding layer
num_words = len(vocab_dict) + 1


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    embeddings_index = {}
    f = open(gloveFile, encoding='utf8')
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


glove_model = loadGloveModel('glove.6B.300d.txt')


def build_glove_embedding_layers():
    embed_matrix = np.zeros((max_features, embedding_dims))
    for word, indx in tokenizer.word_index.items():
        if indx >= max_features:
            continue
        if word in glove_model:
            embed_vec = glove_model[word]
            if embed_vec is not None:
                embed_matrix[indx] = embed_vec
    return embed_matrix


embedding_weights = build_glove_embedding_layers()


### data preprocessing
def preprocessing_txt_keras(statement):
    txt = text_to_word_sequence(statement)
    val = [0] * 32
    val = [vocab_dict[t] for t in txt if t in vocab_dict]  ##replace unknown words with zero index
    return val


## training instances list


nltk.download('stopwords')
from nltk.corpus import stopwords


stop = set(stopwords.words('english'))

### remove stopwords first
train_data['Statement'] = list(
    map(' '.join, train_data['Statement'].apply(lambda x: [item for item in x.lower().split() if item not in stop])))
test_data['Statement'] = list(
    map(' '.join, test_data['Statement'].apply(lambda x: [item for item in x.lower().split() if item not in stop])))
val_data['Statement'] = list(
    map(' '.join, val_data['Statement'].apply(lambda x: [item for item in x.lower().split() if item not in stop])))

train_data['word_id'] = train_data['Statement'].apply(preprocessing_txt_keras)
test_data['word_id'] = test_data['Statement'].apply(preprocessing_txt_keras)
val_data['word_id'] = val_data['Statement'].apply(preprocessing_txt_keras)

x_train = train_data['word_id']
x_test = test_data['word_id']
x_val = val_data['word_id']

y_train = train_data['multi_label']
y_val = val_data['multi_label']


x_train = sequence.pad_sequences(x_train, maxlen=num_steps, padding='post', truncating='post')
y_train = to_categorical(y_train, num_classes=6)
x_val = sequence.pad_sequences(x_val, maxlen=num_steps, padding='post', truncating='post')
y_val = to_categorical(y_val, num_classes=6)
x_test = sequence.pad_sequences(x_test, maxlen=num_steps, padding='post', truncating='post')

## meta data preparation
tr_party = to_categorical(train_data['party_id'], num_classes=num_party)
tr_state = to_categorical(train_data['state_id'], num_classes=num_state)
tr_cont = to_categorical(train_data['context_id'], num_classes=num_context)
tr_job = to_categorical(train_data['job_id'], num_classes=num_job)
tr_subj = to_categorical(train_data['subject_id'], num_classes=num_sub)
tr_speaker=to_categorical(train_data['speaker_id'], num_classes=num_speaker)



# ## put all metadata of train data together in one stack
x_train_metadata_all= np.hstack((tr_party, tr_state, tr_job, tr_subj, tr_speaker, tr_cont))

#### Meta data options by Uzoamaka Ezeakunne ###
x_train_metadata_subj= tr_subj
x_train_metadata_speaker= tr_speaker
x_train_metadata_job= tr_job
x_train_metadata_state= tr_state
x_train_metadata_party= tr_party
x_train_metadata_cont= tr_cont





# #********************************************************************************#
val_party = to_categorical(val_data['party_id'], num_classes=num_party)
val_state = to_categorical(val_data['state_id'], num_classes=num_state)
val_cont = to_categorical(val_data['context_id'], num_classes=num_context)
val_job = to_categorical(val_data['job_id'], num_classes=num_job)
val_subj = to_categorical(val_data['subject_id'], num_classes=num_sub)
val_speaker=to_categorical(val_data['speaker_id'], num_classes=num_speaker)


# ## put all metadata of train data together in one stack
x_val_metadata_all= np.hstack((val_party, val_state, val_job, val_subj, val_speaker, val_cont))

#### Meta data options by Uzoamaka Ezeakunne ###
x_val_metadata_subj= val_subj
x_val_metadata_speaker= val_speaker
x_val_metadata_job= val_job
x_val_metadata_state= val_state
x_val_metadata_party= val_party
x_val_metadata_cont= val_cont


# #********************************************************************************#
te_party = to_categorical(test_data['party_id'], num_classes=num_party)
te_state = to_categorical(test_data['state_id'], num_classes=num_state)
te_cont = to_categorical(test_data['context_id'], num_classes=num_context)
te_job = to_categorical(test_data['job_id'], num_classes=num_job)
te_subj = to_categorical(test_data['subject_id'], num_classes=num_sub)
te_speaker=to_categorical(test_data['speaker_id'], num_classes=num_speaker)

# ## put all metadata of train data together in one stack
x_test_metadata_all= np.hstack((te_party, te_state, te_job, te_subj, te_speaker, te_cont))

#### Meta data options by Uzoamaka Ezeakunne ###
x_test_metadata_subj= te_subj
x_test_metadata_speaker= te_speaker
x_test_metadata_job= te_job
x_test_metadata_state= te_state
x_test_metadata_party= te_party
x_test_metadata_cont= te_cont




# embedding
statement_input = Input(shape=(num_steps,), dtype='int32', name='input')
embed_sequences = Embedding(vocab_length + 1, embedding_dims, weights=[embedding_weights], input_length=num_steps,
                            trainable=False)(statement_input)  # Preloaded glove embeddings



#################                   END OF DATA PRE-PROCESSING          ###############################




# -------------- HYBRID CNN MODEL with Hyperparameters and Visualization !!! ------------


# set meta data (change last word to speaker, subj, job, state, party or cont).
x_train_metadata=x_train_metadata_speaker
x_val_metadata=x_val_metadata_speaker
x_test_metadata=x_test_metadata_speaker

print(x_train_metadata.shape)


# Hyperparameters after tuning
filter_sizes = [3, 4, 5]
num_filters = 128
batch_size = 400
array_of_filters = []
dropout_value1 = 0.5
dropout_value2 = 0.8
num_epochs = 10
lr = 0.01
decay = 1e-6
momentum =0.99

for filter in filter_sizes:
    mod = Conv1D(filters=num_filters, kernel_size=filter,
                 activation="relu", strides=1)(embed_sequences)
    mod = MaxPool1D(3)(mod)
    flatten_mod = Flatten()(mod)
    drop_mod = Dropout(dropout_value1)(flatten_mod)
    array_of_filters.append(drop_mod)

convolut = keras.layers.concatenate(array_of_filters)
convolut = Dense(num_filters, activation='relu')(convolut)

# convolutional layers
convolut_layer_1 = Conv1D(num_filters, 3, activation="relu", padding="valid", strides=1)(embed_sequences)
pooling_layer_1 = MaxPooling1D(3)(convolut_layer_1)
convolut_layer_2 = Conv1D(num_filters, 3, activation="relu")(pooling_layer_1)
pooling_layer_2 = MaxPooling1D(3)(convolut_layer_2)
flatten_layer = Flatten()(pooling_layer_2)
convolut1 = Dropout(dropout_value2)(flatten_layer)
convolut1 = Dense(num_filters, activation='relu')(convolut1)

# merge
merged_convolut = keras.layers.concatenate([convolut, convolut1])

# input for meta data
meta_input = Input(shape=(x_train_metadata.shape[1],), name='meta_input')
drop_mod = Dropout(dropout_value1)(meta_input)
meta_mod = Dense(batch_size, activation='relu')(drop_mod)
mod = keras.layers.concatenate([merged_convolut, meta_mod])

main_output = Dense(6, activation='softmax', name='output')(mod)
model = Model(inputs=[statement_input, meta_input], outputs=[main_output])



model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# fit the model
history = model.fit({'input': x_train, 'meta_input': x_train_metadata},
        {'output': y_train}, epochs=num_epochs, batch_size=batch_size,
        validation_data=({'input': x_val, 'meta_input': x_val_metadata},
                         {'output': y_val} ), callbacks=[es])






#######     VISUALIZATION OF RESULT     ################


plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Accuracy vs Epochs')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()