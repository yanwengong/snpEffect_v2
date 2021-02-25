from config.config import Config
from config.config_trans import ConfigTrans # TODO how to use Config_trans here?
import json

class Utils:

    @staticmethod
    def read_json(file_path, step):
        """ Read json to dict. """
        with open(file_path, 'rt') as f:
            if step == "main":
                return json.load(f, object_hook=lambda a: Config(**a))
            elif step == "transfer_learning":
                return json.load(f, object_hook=lambda a: ConfigTrans(**a))

if __name__ == '__main__':
    config = Utils.read_json('../config/config_0.json')
    print(config.input_train_data_path)
    print(config.input_test_data_path)
    print(config.output_evaluation_data_path)
    print(config.model_name)
    print(config.model_path)
    print(config.subset)


