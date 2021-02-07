from v2.model.danq import DanQ

class Config:
    def __init__(self,
                 input_train_data_path,
                 input_test_data_path,
                 model_name,
                 model_path,
                 output_evaluation_data_path,
                 subset,
                 exact_divide):

        self._model_name = model_name
        self._model_path = model_path
        self._input_train_data_path = input_train_data_path
        self._input_test_data_path = input_test_data_path
        self._output_evaluation_data_path = output_evaluation_data_path
        self._subset = subset
        self._exact_divide = exact_divide

    @property
    def input_train_data_path(self):
        return self.input_train_data_path

    @property
    def input_test_data_path(self):
        return self.input_test_data_path

    @property
    def model(self):
        if self._model_name == "DanQ":
            return DanQ
        else:
            return

    @property
    def output_evaluation_data_path(self):
        return self.output_evaluation_data_path

    @property
    def model_path(self):
        return self.model_path

    @property
    def subset(self):
        return self._subset

    @property
    def exact_divide(self):
        return self._exact_divide