from config.config import Config
import json

class Utils:

    @staticmethod
    def read_json(file_path):
        """ Read json to dict. """
        with open(file_path, 'rt') as f:
            return json.load(f, object_hook=lambda a: Config(**a))

if __name__ == '__main__':
    config = Utils.read_json('../config/config_0.json')
    print(config.input_train_data_path)
    print(config.input_test_data_path)
    print(config.output_evaluation_data_path)
    print(config.model_name)
    print(config.model_path)
    print(config.subset)


