from model.danq import DanQ, simple_DanQ


class Config():

    def __init__(self,
                 X_train_data_path,
                 X_test_data_path,
                 y_train_data_path,
                 y_test_data_path,
                 model_name,
                 num_epochs,
                 batch_size,
                 learning_rate,
                 weight_decay,
                 model_path,
                 output_evaluation_data_path,
                 subset,
                 cell_cluster):

        self.X_train_data_path = X_train_data_path
        self.X_test_data_path = X_test_data_path
        self.y_train_data_path = y_train_data_path
        self.y_test_data_path = y_test_data_path
        self._model_name = model_name
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._model_path = model_path
        self._output_evaluation_data_path = output_evaluation_data_path
        self._subset = subset
        self._cell_cluster = cell_cluster
        self._n_class = len(cell_cluster)

    @property
    def input_train_data_path(self):
        return self._input_train_data_path

    # @input_train_data_path.setter
    # def input_train_data_path(self, input_train_data_path):
    #     self._input_train_data_path = input_train_data_path

    @property
    def input_test_data_path(self):
        return self._input_test_data_path

    # @input_test_data_path.setter
    # def input_test_data_path(self, input_test_data_path):
    #     self._input_test_data_path = input_test_data_path

    @property
    def output_evaluation_data_path(self):
        return self._output_evaluation_data_path

    # @output_evaluation_data_path.setter
    # def output_evaluation_data_path(self, output_evaluation_data_path):
    #     self._output_evaluation_data_path = output_evaluation_data_path

    @property
    def model_name(self):
        return self._model_name

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def weight_decay(self):
        return self._weight_decay

    # @model_name.setter
    # def model_name(self, model_name):
    #     self._model_name = model_name

    @property
    def model_path(self):
        return self._model_path

    # @model_path.setter
    # def model_path(self, model_path):
    #     self._model_path = model_path

    @property
    def subset(self):
        return self._subset

    @property
    def cell_cluster(self):
        return self._cell_cluster

    @property
    def n_class(self):
        return self._n_class

    # @subset.setter
    # def subset(self, subset):
    #     self._subset = subset

    def get_model(self):
        if self._model_name == "DanQ":
            return DanQ(self.n_class)
        elif self._model_name == "simple_DanQ":
            return simple_DanQ(self.n_class )
        else:
            return
