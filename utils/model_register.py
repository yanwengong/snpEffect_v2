from model.danq import DanQ, Simple_DanQ, Complex_DanQ
from model.chvon import Chvon2, Chvon3
from model.deepatt import DeepATT, DeepATT_modified
from model.cnn_activation import CnnActivation
from model.deepsea import DeepSEA, DeepSEA2, DeepSEA3, DeepSEA4
from model.decode import Decode, Decode2
from model.block import BasicBlock


class Model_Register():

    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self, n_class):
        if self.model_name == "DanQ":
            return DanQ(n_class)
        elif self.model_name == "simple_DanQ":
            return Simple_DanQ(n_class)
        elif self.model_name == "complex_DanQ":
            print("complex_DanQ model")
            return Complex_DanQ(n_class)
        elif self.model_name == "Chvon2":
            print("Chvon2 model")
            return Chvon2(n_class)
        elif self.model_name == "Chvon3":
            print("Chvon3 model")
            return Chvon3(n_class)
        elif self.model_name == "DeepATT":
            print("DeepATT model")
            # query = Query(n_class, batch_size).get_query_tensor()
            return DeepATT(n_class)
        elif self.model_name == "DeepATT_modified":
            print("DeepATT modified")
            return DeepATT_modified(n_class)
        elif self.model_name == "CnnActivation":
            print("CnnActivation")
            return CnnActivation(n_class)
        elif self.model_name == "DeepSea":
            print("DeepSea")
            return DeepSEA(n_class)
        elif self.model_name == "DeepSea2":
            print("DeepSea2")
            return DeepSEA2(n_class)
        elif self.model_name == "DeepSea3":
            print("DeepSea3")
            return DeepSEA3(n_class)
        elif self.model_name == "DeepSea4":
            print("DeepSea4")
            return DeepSEA4(n_class)
        elif self.model_name == "Decode":
            print("Decode")
            return Decode(n_class)
        elif self.model_name == "Decode2":
            print("Decode2")
            return Decode2(n_class)
        elif self.model_name == "BasicBlock":
            print("BasicBlock")
            return BasicBlock(n_class)
        else:
            print("--------No Model Retrieved!-------")
            return
