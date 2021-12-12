import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import os
import numpy as np
from model.deepatt import ChQueryDiagonal


class Trainer:
    def __init__(self, registered_model, train_data, eval_data, model_path, num_epochs, batch_size,
                 learning_rate, weight_decay, use_pos_weight, pos_weight, plot_path, n_class, n_class_trans,
                 load_trans_model, trans_model_path):

        '''
        :param model: the model
        :param train_data: complete train data, include x and y
        :param model_path: path the save the trained model
        '''

        self.registered_model = registered_model
        self.train_data = train_data
        self.eval_data = eval_data
        self._model_path = model_path
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.use_pos_weight = use_pos_weight
        self.pos_weight = pos_weight
        self.plot_path = plot_path
        self.n_class = n_class
        self.n_class_trans = n_class_trans
        self.load_trans_model = load_trans_model
        self.trans_model_path = trans_model_path

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

        # check whether to do transfer learning
        if self.load_trans_model == "True":

            model = self.registered_model.get_model(self.n_class_trans)
            # load the previously saved weight
            model.load_state_dict(torch.load(self.trans_model_path))
            # change the output layer dimension
            ## complex DQ
            # num_ftrs = model.Linear2.out_features
            # print("-----------output for the linear2 layer from preload model-------")
            # print(num_ftrs)
            # model.Linear3 = nn.Linear(num_ftrs, self.n_class).to(device)
            # Chvon or DanQ or DeepSEA3
            num_ftrs = model.Linear1.out_features
            print("-----------output for the linear1 layer from preload model-------")
            print(num_ftrs)
            model.Linear2 = nn.Linear(num_ftrs, self.n_class).to(device)

            # deepATT TODO: this is a bit complicated
            # num_out_ftrs = model.multi_head_attention.wq.out_features
            # print("-----------out feature of multi_head_attention.wq -------")
            # print(num_out_ftrs)
            # model.multi_head_attention.wq = nn.Linear(self.n_class, num_out_ftrs, bias=True).to(device)
            # model.category_encoding = ChQueryDiagonal(self.n_class)

            # # TODO check this is right
            print("----------- childern node count --------------")
            ct = 0  # total there are 9 chid for complexDQ
            # total there are 6 chid for DQ
            # total there are 8 child for chvon2
            for child in model.children():
                print(" child", ct, "is:")
                print(child)

                # if ct < 3: # freeze the first two conv
                #     print("child ", ct, " was frozen")
                #     for param in child.parameters():
                #         param.requires_grad = False

                ct += 1
            print("total number of children is: ", ct)

        else:
            print("-----------no preload model loaded, load new model-------")
            model = self.registered_model.get_model(self.n_class)

        model = model.to(device)

        # seed = 1
        # torch.cuda.manual_seed(seed)

        #model.apply(self._weights_init_uniform_rule) # TODO Uniform Initialization added on 03/03

        print("----------Current Model's state_dict-----------")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print("----------Current Model's weight-----------")
        # for param in model.parameters():
        #     print(param.data[0])


        train_loss_hist = []
        eval_loss_hist = []

        train_acc_hist = []
        eval_acc_hist = []

        with open(os.path.join(self.plot_path, "train_vali_record.txt"), 'w') as file:
            file.write("Epoch \t Data \t Loss \t Acc \n")
            file.close()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1000

        if self.use_pos_weight == "False":
            print("not use pos weight in loss")
            #criterion = nn.BCELoss().to(device)
            criterion = nn.BCEWithLogitsLoss()
        elif self.use_pos_weight == "True":
            print("use pos weight in loss")
            #criterion = nn.BCELoss(weight=torch.tensor([self.weight])).to(device) ## for cluster 1(none dominant)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]).to(device)) # TODO, compute weight
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate,
                                     weight_decay=self._weight_decay, amsgrad=True)
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self._batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(self.eval_data, batch_size=self._batch_size, shuffle=True)

        for epoch in range(self._num_epochs):
            print('Epoch {}/{}'.format(epoch, self._num_epochs - 1))
            print('-' * 10)

            # Train
            train_loss = 0.0
            train_acc = 0.0

            model.train()  # set to train mode, use dropout and batchnorm ##

            #for X, y in tqdm(train_loader):
            for train_step, (X, y) in enumerate(train_loader):

                X = X.to(device)
                y = y.to(device)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred_logit = model(X.float()) # predicted value
                #_, preds = torch.max(y_pred, 1) # TODO understand this line

                # Compute and print loss
                # print("---------------y shape----------")
                # print(y_pred_logit.float().shape) # wrong
                # print(y.float().shape)# correct

                loss = criterion(y_pred_logit.float(), y.float())

                # a = list(model.parameters())[0].clone()# TODO modified 03/09, check the weight change before and after loss update
                # Backward and optimize
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad() # clears old gradients from the last step
                loss.backward() # for each parameter, calculate d(loss)/d(weight)
                optimizer.step() # update weights, causes the optimizer to take a step based on the gradients of the parameters

                # statistics
                y_pred_prob = 1 / (1 + np.exp(-y_pred_logit.cpu().detach().numpy())) # calculate prob from logit
                y_pred = y_pred_prob.round() ## 0./1.
                y = y.cpu().detach().numpy()

                train_loss += loss.item() * X.size(0) #loss.item() has be reducted by batch size
                if epoch == 0 and train_step == 0:
                    initial_loss = loss.item()
                    initial_acc = np.sum(y_pred == y)/float(self._batch_size*self.n_class)

                train_acc += np.sum(y_pred == y)

            train_loss = train_loss/len(train_loader.dataset)
            train_acc = train_acc/ float(len(train_loader.dataset)*self.n_class)
            print('Epoch {} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, 'train', train_loss, train_acc))

            with open(os.path.join(self.plot_path, "train_vali_record.txt"), 'a') as file:
                if epoch == 0:
                    file.write("{} \t {} \t {:.4f} \t {:.4f} \n".format(0, 'train', initial_loss, initial_acc))
                    train_loss_hist.append(initial_loss)
                    train_acc_hist.append(initial_acc)
                file.write("{} \t {} \t {:.4f} \t {:.4f} \n".format(epoch+1, 'train', train_loss, train_acc))
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
                    y_pred_logit = model(X.float()) #TODO check whether the weights updated here since optimizer.step()
                    eva_loss = criterion(y_pred_logit.float(), y.float())

                    y_pred_prob = 1 / (1 + np.exp(-y_pred_logit.cpu().detach().numpy()))  # calculate prob from logit
                    y_pred = y_pred_prob.round() ## 0./1.
                    y = y.cpu().detach().numpy()

                    # statistics
                    eval_loss += eva_loss.item() * X.size(0)
                    eval_acc += np.sum(y_pred == y)

                eval_loss = eval_loss / len(eval_loader.dataset)
                eval_acc = eval_acc / float(len(eval_loader.dataset)*self.n_class)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format("validation", eval_loss, eval_acc))
                with open(os.path.join(self.plot_path, "train_vali_record.txt"), 'a') as file:
                    if epoch == 0:
                        # add the fake validation initial loss and acc for the plotting purpose
                        file.write("{} \t {} \t {:.4f} \t {:.4f} \n".format(0, 'validation', initial_loss, initial_acc))
                        eval_loss_hist.append(initial_loss)
                        eval_acc_hist.append(initial_acc)
                    file.write("{} \t {} \t{:.4f} \t {:.4f} \n".format(epoch+1, 'validation', eval_loss, eval_acc))
                    file.close()


            eval_loss_hist.append(eval_loss)
            eval_acc_hist.append(eval_acc)

            if eval_loss < best_loss: # TODO changed on 0318 save the last model based on loss
                best_loss = eval_loss
                best_model_wts = copy.deepcopy(model.state_dict())


        # load best model weights
        # model.load_state_dict(best_model_wts)

        # torch.save(model.state_dict(), self._model_path) # TODO modified 03/09
        torch.save(best_model_wts, self._model_path)
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
        if classname.find('Conv') != -1:
            print("Conv")
            nn.init.xavier_uniform_(m.weight.data)
        # for every Linear layer in a model..
        elif classname.find('Linear') != -1:
            print("linear")
            # get the number of the inputs
            #n = m.in_features
            #y = 1.0 / np.sqrt(n)
            #m.weight.data.uniform_(-y, y)

            m.weight.data.fill_(0.01)
            m.bias.data.fill_(0)


