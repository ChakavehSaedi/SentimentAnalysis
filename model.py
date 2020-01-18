import random
from sklearn import svm
from sklearn import metrics

import torch
#print("torch version", torch.__version__)
import torch.nn as nn
import torch.optim as optim

class svm_handler(nn.Module):
    def __init__(self, log_file_path, svm_kernel):
        super(svm_handler, self).__init__()

        print(" * Building a SVM classifier with %s kernel"%(svm_kernel))
        with open(log_file_path, "a") as log:
            log.write(" * Building a SVM classifier with %s kernel\n"%(svm_kernel))

        if svm_kernel == 'linear':
            self.clf = svm.SVC(kernel='linear')
        else:
            self.clf = svm.SVC(kernel='rbf', gamma=0.7, C=0.9)

    def svm_train(self, train_data, val_data, log_file_path):
        train_inputs = train_data["svm"]
        train_targets = train_data["label"]

        print(" * Training the SVM")
        with open(log_file_path, "a") as log:
            log.write(" * Training the SVM\n")

        self.clf.fit(train_inputs, train_targets)

        print("    - Evaluating the SVM on the validation set")
        val_acc, val_f1 = self.svm_test(val_data)
        print("    - Validation [accuracy %4.3f \t f1 %4.3f]"%(val_acc, val_f1))

        with open(log_file_path, "a") as log:
            log.write("    - Evaluating the SVM on the validation set\n")
            log.write("    - validation [accuracy %4.3f \t f1 %4.3f]\n"%(val_acc, val_f1))

        return val_acc, val_f1

    def svm_test(self, data):
        inputs = data["svm"]
        targets = data["label"]

        false_list = {'fp': [], 'fn': []}

        fp = 0
        fn = 0
        tp = 0
        tn = 0

        y_pred = self.clf.predict(inputs)
        svm_acc = metrics.accuracy_score(targets, y_pred)

        for j, p in enumerate(y_pred):
            if p == targets[j] and p == 1:
                tp += 1
            elif p == targets[j] and p == 0:
                tn += 1
            elif p != targets[j] and p == 1:
                fp += 1
                false_list['fp'].append(data['src'][j])
            elif p != targets[j] and p == 0:
                fn += 1
                false_list['fn'].append(data['src'][j])

        # precision, recall and f1
        fp /= len(y_pred)
        fn /= len(y_pred)
        tp /= len(y_pred)
        tn /= len(y_pred)

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2 * precision * recall / (precision + recall)

        return svm_acc, f1

    def svm_eval(self, str_list):
        pred = self.clf.predict(str_list)

        ans = ["pos" if itm == 1 else "neg" for itm in pred]
        return ans

class net_handler(nn.Module):
    def __init__(self, net_param, verbos, log_file_path):
        super(net_handler, self).__init__()
        self.verbos = verbos
        self.batch_size = net_param['batch_size']
        self.epochs = net_param['epochs']

        torch.manual_seed(11)

        CUDA_LAUNCH_BLOCKING = 1
        CUDA_MAX_THREADS = 32

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.cuda = True
        else:
            self.cuda = False

        ##########################################################
        # neural net initialization
        ##########################################################

        print(" * Building the network")
        with open(log_file_path, "a") as log:
            log.write(" * Building the neural network\n")
        self.net = neural_network(net_param, self.cuda)

        self.net_optimizer = optim.Adam(self.net.parameters(), lr=net_param["l_rate"], betas=(0.9, 0.999))

        self.net_criterion = nn.MSELoss()
        if self.cuda:
            self.net = self.net.to('cuda')
            self.net_criterion = self.net_criterion.to('cuda')

    def net_train(self, train_data, val_data, log_file_path, out_path):
        print(" * Training the network: number of epochs: %d, batch size: %d"%(self.epochs, self.batch_size))
        with open(log_file_path, "a") as log:
            log.write(" * Training the network, Epochs: %d, Batch size: %d\n"%(self.epochs, self.batch_size))

        train_result = {'train': {'loss':[],'acc':[]}, 'validation': {'loss':[],'acc':[], 'f1':[]}}

        for epoch in range(self.epochs):

            train_loss = 0
            train_acc = 0

            for param in self.net.parameters():
                param.requires_grad = True

            ##########################################################
            # prepare batches
            ##########################################################
            train_inputs = train_data["net_vec"]
            train_targets = train_data["label"]

            # shuffle the training data
            c = list(zip(train_inputs, train_targets))
            random.shuffle(c)

            train_inputs, train_targets = zip(*c)

            # extract start and end index of each batch
            batch_indx = self.batchify(len(train_targets), self.batch_size)

            ##########################################################
            # train
            ##########################################################

            for i in range(len(batch_indx)):
                st, ed = batch_indx[i][0], batch_indx[i][1]
                net_input = train_inputs[st:ed]
                net_input = self.tensor_maker(net_input)
                net_target = self.tensor_maker(train_targets[st:ed], "float")

                prediction = self.net(net_input)
                batch_loss = self.net_criterion(prediction, net_target)

                self.net.zero_grad()
                self.net_optimizer.zero_grad()
                batch_loss.backward()

                self.net_optimizer.step()

                # acc
                output = (prediction > 0.5).float()
                correct = (output == net_target).float().sum()
                batch_acc = correct / net_target.shape[0]

                train_loss += batch_loss.item()
                train_acc += batch_acc.item()

                if self.verbos:
                    print("    - Epoch %2d, batch %3d => \t loss: %4.3f\tacc: %4.3f"%(epoch+1, i+1, batch_loss.item(), batch_acc.item()))
                    with open(log_file_path, "a") as log:
                        log.write("    - Epoch %2d, batch %3d => \t loss: %4.3f\tacc: %4.3f\n"%(epoch+1, i+1, batch_loss.item(), batch_acc.item()))

            ##########################################################
            # validation
            ##########################################################
            val_loss, val_acc, val_f1 = self.test_net(val_data)

            train_loss /= len(batch_indx)
            train_acc /= len(batch_indx)

            print("    - Epoch %2d => train [loss: %4.3f \t acc: %4.3f]  -  validation [loss: %4.3f \t acc: %4.3f \t f1: %4.3f]" %
                  (epoch + 1, train_loss, batch_acc, val_loss, val_acc, val_f1))
            with open(log_file_path, "a") as log:
                log.write("    - Epoch %2d => train [loss: %4.3f \t acc: %4.3f]  -  validation [loss: %4.3f \t acc: %4.3f \t f1: %4.3f]\n" %
                  (epoch + 1, train_loss, batch_acc, val_loss, val_acc, val_f1))

            train_result['train']['loss'].append(train_loss)
            train_result['train']['acc'].append(train_acc)
            train_result['validation']['loss'].append(val_loss)
            train_result['validation']['acc'].append(val_acc)
            train_result['validation']['f1'].append(val_f1)

        self.save_net(out_path+'net.h5')

        return train_result

    def test_net(self, data):
        loss = 0
        acc = 0

        fp = 0
        fn = 0
        tp = 0
        tn = 0

        false_list = {'fp':[], 'fn':[]}

        for param in self.net.parameters():
            param.requires_grad = False

        input_data = data["net_vec"]
        target_data = data["label"]

        # extract start and end index of each batch
        batch_indx = self.batchify(len(target_data), self.batch_size)

        for i in range(len(batch_indx)):
            st, ed = batch_indx[i][0], batch_indx[i][1]

            net_input = input_data[st:ed]
            net_input = self.tensor_maker(net_input)
            net_target = self.tensor_maker(target_data[st:ed], "float")

            prediction = self.net(net_input)
            batch_loss = self.net_criterion(prediction, net_target)

            # acc
            output = (prediction > 0.5).float()
            correct = (output == net_target).float().sum()
            batch_acc = correct / net_target.shape[0]

            loss += batch_loss.item()
            acc += batch_acc.item()

            for j, p in enumerate(output):
                if p == net_target[j] and p == 1:
                    tp +=1
                elif p == net_target[j] and p == 0:
                    tn +=1
                elif p != net_target[j] and p == 1:
                    fp +=1
                    false_list['fp'].append(data['src'][j])
                elif p != net_target[j] and p == 0:
                    fn +=1
                    false_list['fn'].append(data['src'][j])

        # loss and accuracy
        loss /= len(batch_indx)
        acc /= len(batch_indx)

        # precision, recall and f1
        fp /= len(batch_indx)
        fn /= len(batch_indx)
        tp /= len(batch_indx)
        tn /= len(batch_indx)

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2 * precision * recall / (precision + recall)

        return loss, acc, f1

    def test_eval(self, data):
        for param in self.net.parameters():
            param.requires_grad = False

        net_input = self.tensor_maker(data)
        prediction = self.net(net_input)

        if len(data)> 1:
            pred = ['pos' if itm.item() > 0.5 else 'neg' for itm in prediction]
        else:
            pred = ['pos' if prediction.item()>0.5 else 'neg']

        return pred

    def tensor_maker(self, v, type=None):
        if type == "float":
            vec = torch.FloatTensor(v)
        else:
            vec = torch.LongTensor(v)

        if self.cuda:
            return vec.to('cuda')
        return vec

    def batchify(self, length, batch_size):
        # batchify (specifying the indexes for beginning and end of each batch)
        batch_indx = []
        en = 0
        while True:
            st = en
            en = batch_size + st
            if en >= length:
                en = length
                if en > st + batch_size*9//10:  # no use in a small batch
                    batch_indx.append((st, en))
                break
            batch_indx.append((st, en))
        return batch_indx

    def save_net(self, path):
        torch.save(self.net.state_dict(), path)

class neural_network(nn.Module):
    def __init__(self, net_param, cuda):
        super(neural_network, self).__init__()

        self.emb_size = net_param["emb_size"]
        self.cnn = net_param["cnn"]
        self.cnn_num = len(self.cnn)
        self.dense_size = net_param["dense"]

        self.embedding = nn.Embedding(self.emb_size[0], self.emb_size[1])

        cnn_layer_IO = [self.emb_size[1]] + [self.cnn[i][0] for i in range(len(self.cnn))]
        pad = [0, 1, 1, 1]

        self.encoder = nn.Sequential()
        for i in range(1, self.cnn_num + 1):
            layer = nn.Conv1d(in_channels=cnn_layer_IO[i - 1], out_channels=cnn_layer_IO[i],
                              kernel_size=self.cnn[i - 1][1], padding=pad[i - 1])
            self.encoder.add_module("CNN" + str(i), layer)

            # layer = nn.BatchNorm1d(cnn_layer_IO[i])
            # self.encoder.add_module("BN" + str(i), layer)

            layer = nn.ReLU(inplace=True)
            self.encoder.add_module("Relu" + str(i), layer)
            #layer = nn.Tanh()
            #self.encoder.add_module("Tanh" + str(i), layer)

        self.linear = nn.Linear(self.dense_size, 1)
        self.sigmoid = nn.Sigmoid()

        if cuda:
            self.embedding.to('cuda')
            self.encoder.to('cuda')
            self.linear.to('cuda')
            self.sigmoid.to('cuda')

        self.init_weights_normal()
        #self.init_weights_uniform

        # print(self.encoder)

    def init_weights_uniform(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.uniform_(-initrange, initrange)

            if isinstance(m, nn.Linear):
                nn.init.uniform_(-initrange, initrange)
                nn.init.constant_(m.bias, 0)

    def init_weights_normal(self):
        mean = 0.0
        std = 0.02
        self.embedding.weight.data.normal_(mean, std)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0)

    def forward(self, net_input, mode="full"):
        emb_inp = self.embedding(net_input).transpose(1, 2)
        hid_out = self.encoder(emb_inp)
        hid_out, _ = torch.max(hid_out, 2)  # behaves as global maxpooling
        prediction = self.sigmoid(self.linear(hid_out))
        return prediction.squeeze()