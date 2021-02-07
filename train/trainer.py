class Trainer:
    def __init__(self, model, data_loader, model_path):
        self._model = model
        self._data_loader = data_loader
        self._model_path = model_path

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
        x = self._data_loader.get_x()

        y = self._data_loader.get_y()

        # Construct our model by instantiating the class defined above
        # Construct our loss function and an Optimizer. Training this strange model with
        # vanilla stochastic gradient descent is tough, so we use momentum

        # might be configurable from config json
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-8, momentum=0.9)

        for t in range(30000):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self._model(x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            if t % 2000 == 1999:
                print(t, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Result: {self._model}')

        # end
        torch.save(self._model, self._model_path)

