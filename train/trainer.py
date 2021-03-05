import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import os
import numpy as np

class Trainer:
    def __init__(self, model, train_data, eval_data, model_path, num_epochs, batch_size,
                 learning_rate, weight_decay, weight, plot_path):

        '''
        :param model: the model
        :param train_data: complete train data, include x and y
        :param model_path: path the save the trained model
        '''

        self._model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self._model_path = model_path
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.weight = weight
        self.plot_path = plot_path

    # @property
    # def data(self):
    #     return self._data
    #
    # @data.setter
    # def data(self, data):
    #     self._data = data

    def train(self):
        """ train """

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self._model.to(device)

        train_loss_hist = []
        eval_loss_hist = []

        train_acc_hist = []
        eval_acc_hist = []

        with open(os.path.join(self.plot_path, "train_vali_record.txt"), 'w') as file:
            file.write("Epoch \t Data \t Loss \t Acc \n")
            file.close()

        # with open(os.path.join(self.plot_path, "vali_record.txt"), 'w') as file:
        #     file.write("Epoch \t Loss \t Acc \n")
        #     file.close()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        if self.weight == "NA":
            criterion = nn.BCELoss().to(device)
        else:
            criterion = nn.BCELoss(weight = torch.tensor([self.weight])).to(device) ## for cluster 1(none dominant)

        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self._batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(self.eval_data, batch_size=self._batch_size, shuffle=True)

        #model.apply(self._weights_init_uniform_rule) # TODO Uniform Initialization added on 0303

        for epoch in range(self._num_epochs):
            print('Epoch {}/{}'.format(epoch, self._num_epochs - 1))
            print('-' * 10)

            # Train

            train_loss = 0.0
            train_acc = 0.0
            model.train()  # set to train mode, use dropout and batchnorm ##

            for X, y in tqdm(train_loader):

                X = X.to(device)
                y = y.to(device)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred_prob = model(X.float()) # predicted value
                #_, preds = torch.max(y_pred, 1) # TODO understand this line

                # Compute and print loss
                loss = criterion(y_pred_prob.float(), y.float())

                # Backward and optimize
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad() # clears old gradients from the last step
                loss.backward() # for each parameter, calculate d(loss)/d(weight)
                optimizer.step() # update weights, causes the optimizer to take a step based on the gradients of the parameters

                # statistics
                y_pred = (y_pred_prob > 0.5).float()  ## 0/1
                train_loss += loss.item() * X.size(0) #loss.item() has be reducted by batch size

                train_acc += torch.sum(y_pred == y)

            if epoch % 1 == 0:
                train_loss = train_loss/len(train_loader.dataset)
                train_acc = train_acc.double() / len(train_loader.dataset)
                print('Epoch {} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, 'train', train_loss, train_acc))

                with open(os.path.join(self.plot_path, "train_vali_record.txt"), 'a') as file:
                    file.write("{} \t {} \t {:.4f} \t {:.4f} \n".format(epoch, 'train', train_loss, train_acc))
                    file.close()



            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)



            # Evaluation
            eval_loss = 0
            eval_acc = 0

            model.eval() # added to not use drop out and batch norm for validation
            with torch.no_grad(): # disable gradiant calculation
                for X, y in tqdm(eval_loader):
                    optimizer.zero_grad() # make sure training and eval has minimum diff
                    X, y = X.to(device), y.to(device)
                    y_pred_prob = model(X.float()) #TODO check whether the weights updated here since optimizer.step()
                    eva_loss = criterion(y_pred_prob.float(), y.float())

                    y_pred = (y_pred_prob > 0.5).float() ## 0/1

                    # statistics
                    eval_loss += eva_loss.item() * X.size(0)
                    eval_acc += torch.sum(y_pred == y)

                eval_loss = eval_loss / len(eval_loader.dataset)
                eval_acc = eval_acc.double() / len(eval_loader.dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format("validation", eval_loss, eval_acc))
                with open(os.path.join(self.plot_path, "train_vali_record.txt"), 'a') as file:
                    file.write("{} \t {} \t{:.4f} \t {:.4f} \n".format(epoch, 'validation', eval_loss, eval_acc))
                    file.close()

            eval_loss_hist.append(eval_loss)
            eval_acc_hist.append(eval_acc)

            if eval_acc > best_acc:
                best_acc = eval_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        # load best model weights
        model.load_state_dict(best_model_wts)

        torch.save(model.state_dict(), self._model_path)
        self._plot_metrics(train_loss_hist, eval_loss_hist, "loss_history.pdf")
        self._plot_metrics(train_acc_hist, eval_acc_hist, "acc_history.pdf")

    def _plot_metrics(self, train_val, eval_val, plot_name):
        plt.figure()
        plt.plot(train_val, color='green', label="train")
        plt.title('train loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')

        plt.plot(eval_val, color='navy', label="validation")
        plt.title('metrics vs epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(self.plot_path, plot_name))

    def _weights_init_uniform_rule(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)


