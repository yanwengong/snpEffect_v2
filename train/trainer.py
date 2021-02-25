import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, train_data, model_path, num_epochs, batch_size,
                 learning_rate, weight_decay, weight):

        '''
        :param model: the model
        :param train_data: complete train data, include x and y
        :param model_path: path the save the trained model
        '''

        self._model = model
        self.train_data = train_data
        self._model_path = model_path
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.weight = weight

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
        print(device)
        print(torch.cuda.device_count())
        model = self._model.to(device)

        if self.weight == "NA":
            criterion = nn.BCELoss().to(device)
        else:
            criterion = nn.BCELoss(weight = torch.tensor([self.weight])).to(device) ## for cluster 1(none dominant)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self._batch_size, shuffle=True)

        for epoch in range(self._num_epochs):
            for i, (X, y) in enumerate(train_loader):
                X = X.to(device)
                y = y.to(device)
                model.train() # set to train mode, use dropout and batchnorm
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(X.float())

                # Compute and print loss
                loss = criterion(y_pred.float(), y.float())

                # Backward and optimize
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 1 == 0:
                print(epoch, loss.item())

        # end
        torch.save(model, self._model_path)

