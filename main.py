import argparse
import torch
from config.config import Config
from data.data_loader import Data
from train.trainer import Trainer
from evaluate.evaluator import Evaluator
from utils.utils import Utils

if __name__ == '__main__':
    # 1\ Parses the command line arguments and returns as a simple namespace.

    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-e', '--exe_mode', default='train', help='The execution mode.(train/test)')
    parser.add_argument('-c', '--config', default='./config/config_1.json', help='The config file of experiment.')
    #parser.add_argument('-m', '--model_path', default='./model/model_1', help='The path to save model or load the saved model')
    # parser.add_argument('-i', '--identifier', default=0, help='The run id of experiment.')
    # parser.add_argument('-v', '--verbosity', default=0, help='The verbosity of training/testing process.')
    args = parser.parse_args()

    # 2\ Configure the Check the Environment.
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    # tf.debugging.set_log_device_placement(False)
    # tf.config.set_soft_device_placement(True)
    # cpu_devices = tf.config.experimental.list_physical_devices('CPU')
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # if gpu_devices:
    #     for gpu in gpu_devices:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    # print('Check the Deep learning Environment:', flush=True)
    # print('GPU count:{}, Memory growth:{}, Soft device placement:{} ...'.format(len(gpu_devices),True,True), flush=True)

    # 3\ Get the configer.
    config = Utils.read_json(args.config)

    # 4\ Selecting the execution mode.
    if args.exe_mode == 'train':
        # config.set_reproducibility()
        train_data_loader = Data(
            config.X_train_data_path,
            config.y_train_data_path,
            config.cell_cluster,
            config.subset)
        print("--------loader finish------------")

        trainer = Trainer(config.get_model(), train_data_loader, config.model_path)
        trainer.train() # include save model to destination

    elif args.exe_mode == 'test':
        train_data_loader = Data(
            config.X_train_data_path,
            config.y_train_data_path,
            config.subset)

        test_data_loader = Data(
            config.X_test_data_path,
            config.y_test_data_path,
            config.subset)

        #tester = Tester(config.model_path, test_data_loader)
        #tester.test()

        evaluator = Evaluator(train_data_loader, test_data_loader, config.model_path, config.output_evaluation_data_path)
        evaluator.evaluate()

    else:
        print('No mode named {}.'.format(args.exe_mode))