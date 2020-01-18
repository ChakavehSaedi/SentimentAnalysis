import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas

class Tokenizer():
    def __init__(self, tokenizer_info):
        super(Tokenizer, self).__init__()
        self.split = ' '
        self.filters = []
        self.char_level = tokenizer_info['char_level']
        self.no_punc = tokenizer_info['no_punc']
        self.lemmatize = tokenizer_info['lemma']
        self.max_wcount = tokenizer_info['vocab_size']
        self.svm_size = tokenizer_info['svm_size']
        self.lower = tokenizer_info['lower_case']
        self.fix_entries = [('pad',0),('<UKN>',1)]

        self.svm_features = [[],[],[]]   # [pos, neg, all]

        self.itm2idx = {}
        self.idx2itm = {}
        self.itmcounts = {}
        self.itmcounts_pos = {}
        self.itmcounts_neg = {}
        self.finilized = False
        self.total_count = 0

        self.max_seq_len = '?'

        if self.lemmatize and not self.char_level:
            self.lemmatizer = WordNetLemmatizer()

    def svm_fearure_extract(self, str_list, out_path = None):
        svm = []
        stop_words = []#set(stopwords.words('english'))
        if self.svm_features == [[],[],[]]:   # first time checking for the features from the training set
            # finding the most frequent words in positive and negative sentence separately
            vocab_list_pos = [(word, count) for word, count in self.itmcounts_pos.items()]
            vocab_list_pos.sort(key=lambda x: (x[1], x[0]), reverse=True)

            vocab_list_neg = [(word, count) for word, count in self.itmcounts_neg.items()]
            vocab_list_neg.sort(key=lambda x: (x[1], x[0]), reverse=True)

            only_w_pos = [w for w,c in vocab_list_pos if c != 0 and w not in stop_words]
            only_w_neg = [w for w,c in vocab_list_neg if c != 0 and w not in stop_words]

            intersection = set.intersection(set(only_w_pos), set(only_w_neg))

            summary = []
            for w_com in intersection:
                for w, c in vocab_list_pos:
                    if w == w_com:
                        pos = c
                        break

                for w, c in vocab_list_neg:
                    if w == w_com:
                        neg = c
                        break

                summary.append([w_com, pos, neg])

                if w_com not in stop_words and abs(pos - neg) > 15:
                    if pos > neg:
                        self.svm_features[0].append(w_com)     # among the common words, only the ones with big difference in the repetition are picked
                    else:
                        self.svm_features[1].append(w_com)

            # featurese selected from the positive sentences
            i = 0
            while len(self.svm_features[0]) < self.svm_size // 2:
                if only_w_pos[i] not in self.svm_features[0]:
                    self.svm_features[0].append(only_w_pos[i])
                i+=1

            # features selceted from the negative sentences
            i = 0
            while len(self.svm_features[1]) < self.svm_size // 2:
                if only_w_neg[i] not in self.svm_features[1]:
                    self.svm_features[1].append(only_w_neg[i])
                i += 1

            self.svm_features[2] = self.svm_features[0] + self.svm_features[1]
        if out_path != None:
            df = pandas.DataFrame(data={"word": [summary[i][0] for i in range(len(summary))],
                                        "positive": [summary[i][1] for i in range(len(summary))],
                                        "negative":[summary[i][2] for i in range(len(summary))]})
            df.to_csv(out_path+"pos-neg-cnt.csv", sep=',', index=False)

        for str in str_list:
            if self.lower:
                str = str.lower()

            if self.no_punc:
                for p in string.punctuation:
                    str = str.replace(p,'')

            if not self.char_level:
                elements = filter(None,str.split(self.split))
                str = ""
                for itm in elements:
                    if self.lemmatize:
                        itm = self.lemmatizer.lemmatize(itm)
                        if itm not in stop_words:
                            str += itm + ' '

            # based on the most frequent words all in all
            vec = np.zeros(len(self.most_frequent[:self.svm_size]))
            for i, itm in enumerate(self.most_frequent[:self.svm_size]):
                vec[i] = str.count(itm[0])

            # based on an engineered list of most frequent words in positive and negative sentences
            vec = np.zeros(len(self.svm_features[2]))
            for i, itm in enumerate(self.svm_features[2]):
                vec[i] = str.count(itm)

            svm.append(vec)

        return svm


    def fit_on_texts(self, str_list, label_list):
        # to keep track of all elements seen in training data

        stop_words = set([])    # it is better to keep the stop words for NN
        for i, str in enumerate(str_list):
            if self.lower:
                str = str.lower()

            if self.no_punc:
                for p in string.punctuation:
                    str = str.replace(p,'')

            if self.char_level:
                elements = str
            else:
                elements = filter(None,str.split(self.split))

            for itm in elements:
                if self.lemmatize and not self.char_level:
                    itm = self.lemmatizer.lemmatize(itm)
                if itm not in self.filters and itm not in stop_words:
                    self.add_item(itm, label_list[i])

        return

    def add_item(self, itm, tag):
        # adds itm to dictionary and keep their count
        if itm not in self.itmcounts:
            self.total_count += 1
            self.itmcounts[itm] = 1
            self.itm2idx[itm] = self.total_count

            self.itmcounts_pos[itm] = 0
            self.itmcounts_neg[itm] = 0

            if tag == 1:
                self.itmcounts_pos[itm] += 1
            else:
                self.itmcounts_neg[itm] += 1

        else:
            self.itmcounts[itm] += 1

            if tag == 1:
                self.itmcounts_pos[itm] += 1
            else:
                self.itmcounts_neg[itm] += 1

    def texts_to_sequences(self, str_list):
        # text vectorizer <=> turning a piece of text into a vector including digits

        # picking the most frequent items in the vocab list if there are more than max_vocab_cnt
        stop_words = set([])
        max_len = 0
        if not self.finilized:
            self.finalized_items()

        seq_list = []
        for str in str_list:
            if self.lower:
                str = str.lower()

            if self.no_punc:
                for p in string.punctuation:
                    str = str.replace(p, '')

            if self.char_level:
                elements = str
            else:
                elements = filter(None, str.split(self.split))

            seq = []
            for itm in elements:
                if self.lemmatize and not self.char_level:
                    itm = self.lemmatizer.lemmatize(itm)
                if itm not in stop_words:
                    if itm in self.itm2idx:
                        seq.append(self.itm2idx[itm])
                    else:
                        seq.append(self.itm2idx['<UKN>'])

            seq_list.append(seq)

            if max_len < len(seq):
                max_len = len(seq)

        seq_list = self.pad_seq(seq_list, max_len)

        return seq_list

    def finalized_items(self):
        # to finilize the elements based on the set-ups and what was loaded as train data

        vocab_list = [(word, count) for word, count in self.itmcounts.items()]
        vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
        if self.total_count > self.max_wcount:
            # prune by word number - picking the most frequent words
            vocabulary = [vocab_list[i][0] for i in range(self.max_wcount)]
        else:
            vocabulary = [vocab_list[i][0] for i in range(self.total_count)]

        self.most_frequent = vocab_list

        vocabulary.append('<UKN>')
        self.total_count = 0

        all_words = set(self.itm2idx.keys())
        self.itm2idx = {}

        for itm in self.fix_entries:
            if itm[0] != 'pad':
                self.itm2idx.update({itm[0]: itm[1]})
                self.idx2itm.update({itm[1]: itm[0]})
                self.total_count += 1

        for itm in all_words:
            if itm in vocabulary:
                self.total_count += 1
                self.itm2idx[itm] = self.total_count
                self.idx2itm[self.total_count] = itm
            else:
                del self.itmcounts[itm]

        self.finilized = True

    def pad_seq(self, x, max_length):
        # Makes all the entries of the same length by padding 0 at the end

        padded_x = []
        for i in range(len(x)):
            npad = max_length + 1 - len(x[i])
            if npad >= 0:
                padded_x.append(np.pad(x[i], pad_width=npad, mode='constant', constant_values=0)[npad:])
            else:
                padded_x.append(x[i][:max_length])
        return padded_x

    def find_max_lenght(x):
        max_length = 0
        for i in range(len(x)):
            if len(x[i]) > max_length:
                max_length = len(x[i])

        return max_length

    def item_index(self):
        return self.itm2idx

    def index_item(self):
        return self.idx2itm

    def __len__(self):
        return len(self.itm2idx)