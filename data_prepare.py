from os import listdir
from os.path import isfile, join
import csv
import pickle
import pandas as pd
import numpy as np
import numpy.random as rng

from data_tokenizer import Tokenizer

class data_handler():
    def __init__(self, path_to_train, path_to_test, tokenizer_info, log_path, output_path):
        super(data_handler, self).__init__()
        self.tokenizer = Tokenizer(tokenizer_info)

        ########################
        # Reading train data
        ########################

        print(" * Reading train data")
        with open(log_path, 'a') as log_file:
            log_file.write(" * Reading train data\n")

        self.all_train_data = self.read_file(path_to_train)

        print("    - %d entries loaded "%(len(self.all_train_data['src'])))
        print("    - number of positive cases %d "%(self.all_train_data['label'].count(1)))
        print("    - number of negative cases %d "%(self.all_train_data['label'].count(0)))
        with open(log_path, 'a') as log_file:
            log_file.write("    - %d entries loaded \n"%(len(self.all_train_data['src'])))
            log_file.write("    - number of positive cases %d \n" % (self.all_train_data['label'].count(1)))
            log_file.write("    - number of negative cases %d \n" % (self.all_train_data['label'].count(0)))

        ########################
        # Processing train data
        ########################

        self.tokenizer.fit_on_texts(self.all_train_data['src'], self.all_train_data['label'])
        self.all_train_data['net_vec'] = self.tokenizer.texts_to_sequences(self.all_train_data['src'])
        self.all_train_data['svm'] = self.tokenizer.svm_fearure_extract(self.all_train_data['src'], output_path)

        most_fteq = ""
        for itm in self.tokenizer.most_frequent:
            most_fteq += str(itm) + '\n'
        with open(output_path + 'most-feq.txt', 'w', encoding='utf-8') as f:
            f.write(most_fteq)

        print("    - The top 10 most frequent elements are %s"%(str(self.tokenizer.most_frequent[:10]).replace('[','').replace(']','')))
        print("    - The full list of the top frequent elements is saved in /data/output/most-feq.txt")
        print("    - The full list of intersection between top frequent elements between positive and negative sentences are saved in /data/output/pos-neg-cnt.csv")
        with open(log_path, 'a') as log_file:
            log_file.write("    - The top 10 most frequent elements are %s\n"%(str(self.tokenizer.most_frequent[:10]).replace('[','').replace(']','')))
            log_file.write("    - The full list of the top frequent elements is saved in /data/output/most-feq.txt\n")
            log_file.write("    - The full list of intersection between top frequent elements between positive and negative sentences are saved in /data/output/pos-neg-cnt.csv\n")

        self.vocab_size = len(self.tokenizer.itm2idx)

        ########################
        # Train and  Validation sets
        ########################

        self.train_data = {'src':[], 'label':[], 'net_vec':[], 'svm':[]}
        self.val_data = {'src':[], 'label':[], 'net_vec':[], 'svm':[]}

        total = len(self.all_train_data['src'])
        val_size = total//10
        val_indx = rng.choice(total, val_size, replace=False)
        train_indx = [i for i in range(total) if i not in val_indx]

        self.val_data['src'] = [self.all_train_data['src'][i] for i in val_indx]
        self.val_data['label'] = [self.all_train_data['label'][i] for i in val_indx]
        self.val_data['net_vec'] = [self.all_train_data['net_vec'][i] for i in val_indx]
        self.val_data['svm'] = [self.all_train_data['svm'][i] for i in val_indx]

        self.train_data['src'] = [self.all_train_data['src'][i] for i in train_indx]
        self.train_data['label'] = [self.all_train_data['label'][i] for i in train_indx]
        self.train_data['net_vec'] = [self.all_train_data['net_vec'][i] for i in train_indx]
        self.train_data['svm'] = [self.all_train_data['svm'][i] for i in train_indx]

        ########################
        # Reading and processing test data
        ########################

        print(" * Reading test data")
        with open(log_path, 'a') as log_file:
            log_file.write(" * Reading test data \n")

        self.test_data = self.read_file(path_to_test)
        self.test_data['net_vec'] = self.tokenizer.texts_to_sequences(self.test_data['src'])
        self.test_data['svm'] = self.tokenizer.svm_fearure_extract(self.test_data['src'])

        print("    - %d entries loaded " % (len(self.test_data['src'])))
        print("    - number of positive cases %d " % (self.test_data['label'].count(1)))
        print("    - number of negative cases %d " % (self.test_data['label'].count(0)))
        with open(log_path, 'a') as log_file:
            log_file.write("    - %d entries loaded \n" % (len(self.test_data['src'])))
            log_file.write("    - number of positive cases %d \n" % (self.test_data['label'].count(1)))
            log_file.write("    - number of negative cases %d \n" % (self.test_data['label'].count(0)))


        with open(output_path+'tokenizer.h5', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def read_file(self, path):
        data = {'src':[], 'label':[], 'net_vec':[], 'svm':[]}

        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for f in onlyfiles:
            """with open(path+f) as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    data['src'].append(row[0])
                    data['label'].append(row[1].replace('\n', '').replace(' ', ''))"""
            df = pd.read_csv(path+f, sep='\t', header=None, names=['text', 'label'])
            data['src'] += [itm for itm in df['text'].tolist()]
            data['label'] += [itm for itm in df['label'].tolist()]

        return data