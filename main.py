import argparse
import torch
from data.pre_processor import Concate, ConcateTrans, Processor
from data.data_loader import Data
from train.trainer import Trainer
from evaluate.evaluator import Evaluator
from utils.utils import Utils
from utils.model_register import Model_Register
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    registered_model = Model_Register(config.model_name)

    if config.cell_cluster == "all":
        config.cell_cluster = list(range(0, config.n_class, 1))

    print("create the output folder")
    if not os.path.exists(config.output_evaluation_data_path):
        os.makedirs(config.output_evaluation_data_path)

    # 4\ Load, process and split the data set

    if args.step == "main":
        processor = Concate(config.pos_forward_path, config.encode_path, config.label_path,
                            config.encode_n, config.cell_cluster, config.subset)
        data, label, pos_weight = processor.concate_data()

        data_train, data_eval, data_test, label_train, \
        label_eval, label_test, test_size = processor.split_train_test(data, label)
        #data, label, pos_weight = processor.concate_data()
    elif args.step == "transfer_learning":
        processor = ConcateTrans(config.pos_forward_path, config.neg_forward_path,
                                 config.balance, config.subset)
    #data, label, pos_weight = processor.concate_data()
        data, label, pos_weight = processor.concate_data()

        data_train, data_eval, data_test, label_train, label_eval, label_test = processor.split_train_test(data, label)

    # 4\ Selecting the execution mode.
    if args.exe_mode == 'train':
        print("----------train data loader start--------")
        print(config.cell_cluster)
        train_data_loader = Data(data_train, label_train)

        print("----------train data loader finish--------")
        print("----------eval data loader start--------")

        eval_data_loader = Data(data_eval, label_eval)

        print("----------eval data loader finish--------")


        trainer = Trainer(registered_model, train_data_loader, eval_data_loader, config.model_path,
                          config.num_epochs, config.batch_size, config.learning_rate,
                          config.weight_decay, config.use_pos_weight, pos_weight, config.output_evaluation_data_path,
                          config.n_class, config.n_class_trans, config.load_trans_model, config.trans_model_path)

        print("train start time: ", datetime.now())
        trainer.train() # include save model to destination
        print("train end time: ", datetime.now())

    elif args.exe_mode == 'test':
        print("----------train data loader start--------")
        train_data_loader = Data(data_train, label_train)
        print("----------train data loader done--------")

        print("----------test data loader start--------")
        test_data_loader = Data(data_test, label_test)
        print("----------test data loader done--------")

        print("----------all data loader start--------")

        all_data_loader = Data(data, label)

        print("----------all data loader finish--------")

        evaluator = Evaluator(registered_model, train_data_loader, test_data_loader, all_data_loader, config.model_path,
                              config.output_evaluation_data_path, config.batch_size, test_size,
                              config.n_class)
        evaluator.evaluate()

    else:
        print('No mode named {}.'.format(args.exe_mode))