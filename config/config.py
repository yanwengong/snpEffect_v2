from model.danq import DanQ


class Config():

    def __init__(self,
                 X_train_data_path,
                 X_test_data_path,
                 y_train_data_path,
                 y_test_data_path,
                 model_name,
                 model_path,
                 output_evaluation_data_path,
                 subset,
                 cell_cluster):

        self.X_train_data_path = X_train_data_path
        self.X_test_data_path = X_test_data_path
        self.y_train_data_path = y_train_data_path
        self.y_test_data_path = y_test_data_path
        self._model_name = model_name
        self._model_path = model_path
        self._output_evaluation_data_path = output_evaluation_data_path
        self._subset = subset
        self._cell_cluster = cell_cluster
        self.n_class = len(cell_cluster)

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

    # @subset.setter
    # def subset(self, subset):
    #     self._subset = subset

    def get_model(self):
        if self._model_name == "DanQ":
            return DanQ(self.n_class)
        else:
            return