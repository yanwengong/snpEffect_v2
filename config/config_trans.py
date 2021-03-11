from model.danq import DanQ, Simple_DanQ, Complex_DanQ, Simple_DanQ_noLSTM
from model.sai import Net
from model.conv_only import DeepSea
from model.chvon import Chvon, Chvon2
from model.deepatt import DeepATT

class ConfigTrans():

    def __init__(self,
                 # pos_forward_path,
                 # encode_path,
                 # label_path,
                 # encode_n,
                 pos_forward_path,
                 neg_forward_path,
                 model_name,
                 num_epochs,
                 batch_size,
                 learning_rate,
                 weight_decay,
                 weight_value,
                 model_path,
                 output_evaluation_data_path,
                 subset,
                 balance,
                 cell_cluster):

        # self.pos_forward_path = pos_forward_path
        # self.encode_path = encode_path
        # self.label_path = label_path
        # self.encode_n = encode_n
        self.pos_forward_path = pos_forward_path
        self.neg_forward_path = neg_forward_path
        self._model_name = model_name
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.weight_value = weight_value
        self._model_path = model_path
        self._output_evaluation_data_path = output_evaluation_data_path
        self._subset = subset
        self._cell_cluster = cell_cluster
        self._balance = balance
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
    def balance(self):
        return self._balance

    @property
    def cell_cluster(self):
        return self._cell_cluster

    @property
    def n_class(self):
        return self._n_class

    # @subset.setter
    # def subset(self, subset):
    #     self._subset = subset

    # TODO: move this part out to model folder as a new model registration class
    def get_model(self):
        if self._model_name == "DanQ":
            print("--------MODEL: DanQ-------")
            return DanQ(self.n_class)

        elif self._model_name == "simple_DanQ":
            print("--------MODEL: simple_DanQ-------")
            return Simple_DanQ(self.n_class)

        elif self._model_name == "complex_DanQ":
            print("--------MODEL: complex_DanQ-------")
            return Complex_DanQ(self.n_class)

        elif self._model_name == "Simple_DanQ_noLSTM":
            print("--------MODEL: Simple_DanQ_noLSTM-------")
            return Simple_DanQ_noLSTM(self.n_class)

        elif self._model_name == "Sai":
            print("--------MODEL: Net-------")
            return Net()

        elif self._model_name == "DeepSea":
            print("--------MODEL: DeepSea-------")
            return DeepSea()

        elif self._model_name == "Chvon":
            print("--------MODEL: Chvon-------")
            return Chvon()

        elif self._model_name == "Chvon2":
            print("--------MODEL: Chvon2-------")
            return Chvon2()

        elif self._model_name == "DeepATT":
            print("--------MODEL: DeepATT-------")
            return DeepATT()


        else:
            print("--------No Model Retrieved!-------")
            return