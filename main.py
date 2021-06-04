import os
import gc
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
import torch.optim as optim
import torch.utils.data as torchdata
from sklearn.metrics import accuracy_score

rootPath = './'
dataPath = rootPath + '/data/'
tempPath = rootPath + '/temp/'
logsPath = rootPath + '/logs/'
# hyperparameters.
_shuffle_seed_ = 0
_train_ratio_  = 0.8
_batch_size_   = 256
_learn_rate_   = 0.0002
_hidden_size_  = 18
_hidden_layer_ = 1
_max_epochs_   = 300

# Logger: redirect the stream on screen and to file.
class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def demo():
    # load the numpy data and labels.
    embeds = np.load(os.path.join(dataPath, 'a2ps_func_embeds.npy'), allow_pickle=True)
    labels = np.load(os.path.join(dataPath, 'a2ps_func_compiler_label.npy'), allow_pickle=True)
    # shuffle the data as well as the labels with the same seed.
    np.random.seed(seed=_shuffle_seed_)
    np.random.shuffle(embeds)
    np.random.seed(seed=_shuffle_seed_)
    np.random.shuffle(labels)

    # split the data and labels.
    numTrain = int(_train_ratio_ * len(labels))
    dTrain = embeds[0:numTrain]
    lTrain = labels[0:numTrain]
    dTest = embeds[numTrain:]
    lTest = labels[numTrain:]

    # train RNN model.
    model = BinProRNNTrain(dTrain, lTrain, dTest, lTest, batchsize=_batch_size_, learnRate=_learn_rate_)
    pred, acc = BinProRNNTest(model, dTest, lTest, batchsize=_batch_size_)

    # evaluation.
    OutputEval(pred, lTest, method='BinProvRNN')

    return

class BinProvRNN(nn.Module):
    def __init__(self, embedDim=9, hiddenSize=18, hiddenLayers=1):
        super(BinProvRNN, self).__init__()
        # parameters.
        class_num = 2
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedDim, hidden_size=hiddenSize, num_layers=hiddenLayers, bidirectional=True)
        # Fully-Connected Layer
        self.mlp = Sequential(
            Linear(hiddenSize * hiddenLayers * 2, 9),
            ReLU(),
            Linear(9, 2)
        )
        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x             batch_size * max_length * embedDim
        inputs = x.permute(1, 0, 2)
        # inputs        max_length * batch_size * embedDim
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        # lstm_out      max_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # feature_map   batch_size * (hidden_size * num_layers * direction_num)
        final_out = self.mlp(feature_map)    # batch_size * class_num
        return self.softmax(final_out)      # batch_size * class_num

def BinProRNNTrain(dTrain, lTrain, dTest, lTest, batchsize=64, learnRate=0.001):
    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).float().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xTest = torch.from_numpy(dTest).float().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # get training weights.
    lbTrain = [item for sublist in lTrain.tolist() for item in sublist]
    weights = []
    for lb in range(2):
        weights.append(1 - lbTrain.count(lb) / len(lbTrain))
    lbWeights = torch.FloatTensor(weights).cuda()

    # build the model of recurrent neural network.
    model = BinProvRNN(embedDim=dTrain.shape[2], hiddenSize=_hidden_size_, hiddenLayers=_hidden_layer_)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'[INFO] <BinProRNNTrain> ModelType: BinProRNN.')
    print(f'[INFO] <BinProRNNTrain> Para: TrainNum: {dTrain.shape[0]}, TestNum: {dTest.shape[0]}, TrainRate: {_train_ratio_}.')
    print(f'[INFO] <BinProRNNTrain> Para: EmbedDim: {dTrain.shape[2]}, MaxLen: {dTrain.shape[1]}, HidNodes: {_hidden_size_}, HidLayers: {_hidden_layer_}.')
    print(f'[INFO] <BinProRNNTrain> BatchSize: {_batch_size_}, LearningRate: {_learn_rate_}, MaxEpoch: {_max_epochs_}.')
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)
    # memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(_max_epochs_):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # back propagation.
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        lossTrain /= len(dTrain)
        # train accuracy.
        accTrain = accuracy_score(labels, predictions) * 100

        # testing phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(testloader):
                # data conversion.
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                # forward propagation.
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        # test accuracy.
        accTest = accuracy_score(labels, predictions) * 100
        accList.append(accTest)

        # output information.
        print(f'Epoch: {epoch:03d}, Loss: {lossTrain:.4f}, Train Acc: {accTrain:.4f}, Test Acc: {accTest:.4f}')
        # save the best model.
        if (accList[-1] > max(accList[0:-1])):
            torch.save(model.state_dict(), tempPath + '/model_BinProvRNN.pth')

    # load best model.
    model.load_state_dict(torch.load(tempPath + '/model_BinProvRNN.pth'))
    print('[INFO] <BinProvRNNTrain> Finish training BinProvRNN model. (Best model: ' + tempPath + '/model_BinProvRNN.pth)')

    return model

def BinProRNNTest(model, dTest, lTest, batchsize=64):
    # tensor data processing.
    xTest = torch.from_numpy(dTest).float().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # load the model of recurrent neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # testing phase.
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # forward propagation.
            yhat = model.forward(data)  # get output
            # statistic
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # testing accuracy.
    accuracy = accuracy_score(labels, predictions) * 100
    predictions = [[item] for item in predictions]

    return predictions, accuracy

def OutputEval(predictions, labels, method=''):
    '''
    Output the evaluation results.
    :param predictions: predicted labels. [[0], [1], ...]
    :param labels: ground truth labels. [[1], [1], ...]
    :param method: method name. string
    :return: accuracy - the total accuracy. numeric
             confusion - confusion matrix [[1000, 23], [12, 500]]
    '''

    def Evaluation(predictions, labels):
        '''
        Evaluate the predictions with gold labels, and get accuracy and confusion matrix.
        :param predictions: [0, 1, 0, ...]
        :param labels: [0, 1, 1, ...]
        :return: accuracy - 0~1
                 confusion - [[1000, 23], [12, 500]]
        '''

        # parameter settings.
        D = len(labels)
        cls = 2

        # get confusion matrix.
        confusion = np.zeros((cls, cls))
        for ind in range(D):
            nRow = int(predictions[ind][0])
            nCol = int(labels[ind][0])
            confusion[nRow][nCol] += 1

        # get accuracy.
        accuracy = 0
        for ind in range(cls):
            accuracy += confusion[ind][ind]
        accuracy /= D

        return accuracy, confusion

    # get accuracy and confusion matrix.
    accuracy, confusion = Evaluation(predictions, labels)
    precision = confusion[1][1] / (confusion[1][0] + confusion[1][1]) if (confusion[1][0] + confusion[1][1]) else 0
    recall = confusion[1][1] / (confusion[0][1] + confusion[1][1]) if (confusion[0][1] + confusion[1][1]) else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # output on screen and to file.
    print('       -------------------------------------------')
    print('       method           :  ' +  method) if len(method) else print('', end='')
    print('       accuracy  (ACC)  :  %.3f%%' % (accuracy * 100))
    print('       precision (P)    :  %.3f%%' % (precision * 100))
    print('       recall    (R)    :  %.3f%%' % (recall * 100))
    print('       F1 score  (F1)   :  %.3f' % (F1))
    print('       fall-out  (FPR)  :  %.3f%%' % (confusion[1][0] * 100 / (confusion[1][0] + confusion[0][0])))
    print('       miss rate (FNR)  :  %.3f%%' % (confusion[0][1] * 100 / (confusion[0][1] + confusion[1][1])))
    print('       confusion matrix :      (actual)')
    print('                           Neg         Pos')
    print('       (predicted) Neg     %-5d(TN)   %-5d(FN)' % (confusion[0][0], confusion[0][1]))
    print('                   Pos     %-5d(FP)   %-5d(TP)' % (confusion[1][0], confusion[1][1]))
    print('       -------------------------------------------')

    return accuracy, confusion

if __name__ == '__main__':
    # initialize the log file.
    logfile = 'BinProRNN.log'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    demo()