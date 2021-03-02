import argparse
import torch
from data.pre_processor import Processor, ProcessorTrans
from data.data_loader import Data
from train.trainer import Trainer
from evaluate.evaluator import Evaluator
from utils.utils import Utils
from datetime import datetime
import os

if __name__ == '__main__':
    # 1\ Parses the command line arguments and returns as a simple namespace.

    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-e', '--exe_mode', default='train', help='The execution mode.(train/test)')
    parser.add_argument('-c', '--config', default='./config/config_1.json', help='The config file of experiment.')
    parser.add_argument('-s', '--step', default='transfer_learning', help='The step of model (transfer_learning/main)')

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
    config = Utils.read_json(args.config, args.step)
    if not os.path.exists(config.output_evaluation_data_path):
        os.makedirs(config.output_evaluation_data_path)

    # 4\ Load, process and split the data set

    if args.step == "main":
        processor = Processor(config.pos_forward_path, config.encode_path, config.label_path, config.encode_n)
        data, label = processor.concate_data()
    elif args.step == "transfer_learning":
        processor = ProcessorTrans(config.pos_forward_path, config.neg_forward_path)
        data, label = processor.concate_data()
    data_train, data_eval, data_test, label_train, label_eval, label_test = processor.split_train_test(data, label)

    # 4\ Selecting the execution mode.
    if args.exe_mode == 'train':
        print("--------loader start------------")
        train_data_loader = Data(data_train, label_train, config.cell_cluster, config.subset)
        eval_data_loader = Data(data_eval, label_eval, config.cell_cluster, config.subset)

        print("--------loader finish------------")

        trainer = Trainer(config.get_model(), train_data_loader, eval_data_loader, config.model_path,
                          config.num_epochs, config.batch_size, config.learning_rate,
                          config.weight_decay, config.weight_value, config.output_evaluation_data_path)
        print("train start time: ", datetime.now())
        trainer.train() # include save model to destination
        print("train end time: ", datetime.now())

    elif args.exe_mode == 'test':
        train_data_loader = Data(data_train, label_train, config.cell_cluster, config.subset)

        print("----------train data loader done--------")
        test_data_loader = Data(data_test, label_test, config.cell_cluster, config.subset)

        print("----------test data loader done--------")

        evaluator = Evaluator(config.get_model(), train_data_loader, test_data_loader, config.model_path,
                              config.output_evaluation_data_path, config.batch_size,
                              config.n_class)
        evaluator.evaluate()

    else:
        print('No mode named {}.'.format(args.exe_mode))