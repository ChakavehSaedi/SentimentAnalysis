"""
Supervised Sentiment analysis

Summary
1- Fetchining the train and test data saved in the /data/input/ directory
2- Turning the data into [entry][label] format
3- Turning the data into format readable for a machine learning model
   3-1- SVM using frequent words as features
   3-2 NN using text to sequence to vectorize input text sentences
4- Training the model
5- Testing the model


"""
import os
from datetime import datetime
import argparse
from data_prepare import data_handler
from model import net_handler, svm_handler
#############################################################################
# Arguments
#############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-trp", "--train_path", default="/data/input/train/")
parser.add_argument("-tsp", "--test_path", default="/data/input/test/")
parser.add_argument("-op", "--output_path", default="/data/output/")

parser.add_argument("-lm", "--lemmatize", default=True)
parser.add_argument("-chl", "--char_level", default=False)
parser.add_argument("-lc", "--lower_case", default=True)
parser.add_argument("-rp", "--remove_punctuation", default=True)

parser.add_argument("-vz", "--vocab_size", default=10000)
parser.add_argument("-embs", "--embedding_size", default=300)
parser.add_argument("-cnn", "--cnn_info", default=[[350,1], [250,2], [200,3]])
parser.add_argument("-ds", "--dense_size", default=200)
parser.add_argument("-l_rate", "--learning_rate", default=0.001)
parser.add_argument("-v", "--verbos", default=False)
parser.add_argument("-bs", "--batch_size", default=25)
parser.add_argument("-e", "--epochs", default=5)

parser.add_argument("-svms", "--svm_size", default=3000)
parser.add_argument("-svmk", "--svm_kernel", default='linear')    # linear or rbf


##############################################################################
# variables and parameters
##############################################################################

args = parser.parse_args()

# path
train_path = os.getcwd()+ args.train_path
test_path = os.getcwd() + args.test_path
output_path = os.getcwd() + args.output_path

# tokenizer
tokenizer_info = {'lemma':args.lemmatize, 'char_level': args.char_level, 'lower_case':args.lower_case,
                  'vocab_size': args.vocab_size, 'no_punc': args.remove_punctuation, 'svm_size':args.svm_size}

# network
net_param = {}
net_param["emb_size"] = [-1, args.embedding_size]     # the first element = vocabulary size will be set later
net_param["cnn"] = args.cnn_info
net_param["dense"] = args.dense_size
net_param["l_rate"] = args.learning_rate
net_param["batch_size"] = args.batch_size
net_param["epochs"] = args.epochs

########################
# log info
########################

log_path = output_path + 'log.txt'
print("Sentiment Analysis, %s"%(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
print("===================== Data Prepration ======================")
with open(log_path,'w') as log_file:
    log_file.write("Sentiment Analysis, %s\n"%(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    log_file.write("===================== Data Prepration ======================\n")


#############################
# data prepaer for SVM and NN
#############################

data_process = data_handler(train_path, test_path, tokenizer_info, log_path, output_path)
net_param["emb_size"][0] = data_process.vocab_size + 1

train_data = data_process.train_data
val_data = data_process.val_data
test_data = data_process.test_data

##############################
# SVM model
##############################
print("======================== SVM Model =========================")
with open(log_path, "a") as log:
    log.write("======================== SVM Model =========================\n")

svm = svm_handler(log_path, args.svm_kernel)

# Train the SVM
val_acc, val_f1 = svm.svm_train(train_data, val_data, log_path)

print(" * Testing the SVM")
with open(log_path, "a") as log:
    log.write(" * Testing the SVM\n")
test_acc, test_f1 = svm.svm_test(test_data)

print("    - test result [accuracy %4.3f \t f1 %4.3f]" % (test_acc, test_f1))
with open(log_path, "a") as log:
    log.write("    - test result [accuracy %4.3f \t f1 %4.3f]\n" % (test_acc, test_f1))


##############################
# NN model
##############################
print("========================= NN Model =========================")
with open(log_path, "a") as log:
    log.write("========================= NN Model =========================\n")

nn = net_handler(net_param, args.verbos, log_path)

# Train the network
train_results = nn.net_train(train_data, val_data, log_path, output_path)

print(" * Testing the network")
with open(log_path, "a") as log:
    log.write(" * Testing the network\n")
test_loss, test_acc, test_f1 = nn.test_net(test_data)

print("    - Test result [loss: %4.3f \t acc: %4.3f \t f1: %4.3f]" % (test_loss, test_acc, test_f1))
with open(log_path, "a") as log:
    log.write("    - Test result [loss: %4.3f \t acc: %4.3f \t f1: %4.3f]\n" % (test_loss, test_acc, test_f1))

print("========================= Test the model live =========================")

while True:
    test_string = input("Enter a String to classify, 'end' to stop\n")
    if test_string.lower() == 'end':
        break

    # SVM prediction
    svm_vec = data_process.tokenizer.svm_fearure_extract([test_string])
    svm_result = svm.svm_eval(svm_vec)

    # NN prediction
    nn_vec = data_process.tokenizer.texts_to_sequences([test_string])
    nn_result = nn.test_eval(nn_vec)

    print("    - SVM decision: %s"%(svm_result[0]))
    print("    - NN decision: %s"%(nn_result[0]))

