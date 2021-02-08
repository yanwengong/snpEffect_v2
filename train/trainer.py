import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, train_data, model_path, num_epochs, batch_size,
                 learning_rate, weight_decay):

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

    # @property
    # def data(self):
    #     return self._data
    #
    # @data.setter
    # def data(self, data):
    #     self._data = data

    def train(self):
        """ train """

        # Create Tensors to hold input and outputs.
        #x = self._data_loader.get_x()

        #y = self._data_loader.get_y()

        # Construct our model by instantiating the class defined above
        # Construct our loss function and an Optimizer. Training this strange model with
        # vanilla stochastic gradient descent is tough, so we use momentum


        # num_epochs = 1
        # batch_size = 512
        # learning_rate = 0.001
        # weight_decay = 0.001

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        print(torch.cuda.device_count())
        model = self._model.to(device)


        # TODO: read from json
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self._batch_size, shuffle=True)

        for epoch in range(self._num_epochs):
            for i, (X, y) in enumerate(train_loader):
                X = X.to(device)
                y = y.to(device)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(X.float())

                # Compute and print loss
                loss = criterion(y_pred.float(), y.float())
                if epoch % 10 == 0:
                    print(epoch, loss.item())

                # Backward and optimize
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #print(f'Result: {model}')

        # end
        torch.save(model, self._model_path)

