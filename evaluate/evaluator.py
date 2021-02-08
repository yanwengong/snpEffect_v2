import torch
import os
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
import pandas as pd


class Evaluator:
    def __init__(self, train_data, test_data, model_path, output_evaluation_data_path,
                 batch_size, n_class):
        '''

        :param train_data: train data
        :param test_data: test_data
        :param model_path: path the model is saved
        :param output_evaluation_data_path: path to save all the evaluation result
        '''
        self._train_data = train_data
        self._test_data = test_data
        self._mode = torch.load(model_path)
        self._output_evaluation_data_path = output_evaluation_data_path
        self._batch_size = batch_size
        self.n_class = n_class

    def _predict(self, data_loader):
        self._mode.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():

            y_arr = np.empty((0, self.n_class))
            y_hat_arr = np.empty((0, self.n_class))

            for X, y in data_loader:
                X = X.to(device)
                #images = torch.reshape(images, (images.shape[0], images.shape[2], images.shape[1]))
                y = y.to(device)
                outputs = self._mode(X.float())
                #print('outputs', outputs.shape, outputs)
                # _, pred = torch.max(outputs.data, 1)
                y_hat = torch.round(outputs)
                y = y.float()
                #y.extend(y.tolist())
                y_arr = np.concatenate((y_arr, y.numpy()))
                y_hat_arr = np.concatenate((y_hat_arr, y_hat.numpy()))
                #y_hat_arr.append(y_hat.reshape(-1).tolist())

            print("------------------ evaluation shape --------------------")
            print(y_arr.shape)
            print(y_hat_arr.shape)
            print("------------------ done --------------------")

            return y_arr, y_hat_arr

    def evaluate(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        train_loader = torch.utils.data.DataLoader(self._train_data, batch_size=self._batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self._test_data, batch_size=self._batch_size, shuffle=True)

        y_train, y_train_hat = self._predict(train_loader)
        y_test, y_test_hat = self._predict(test_loader)

        # calculate and save overall performance metrics
        perf_metrics = self.get_overall_performance_metrics(
            y_train, self._format(y_train_hat),
            y_test, self._format(y_test_hat),
            threshold=0.5, average="micro")

        # file_name = "".join(["perf_metrics", ".csv"])
        perf_metrics.to_csv(os.path.join(self._output_evaluation_data_path, "perf_metrics.csv"))


        # plot_name = "".join(["roc", ".pdf"])
        # plot overall roc
        self.plot_overall_roc(
            y_train, y_train_hat,
            y_test, y_test_hat,
            os.path.join(self._output_evaluation_data_path, "roc.pdf"))

        # ploc overall roc and save individual metrics
        self.plot_individual_roc(
            y_train, y_train_hat,
            y_test, y_test_hat,
            self._output_evaluation_data_path)

    def plot_individual_roc(self, y_train, y_train_hat, y_test, y_test_hat, plot_path):
        for i in range(y_train.shape[1]):
            individual_y_train = y_train[:, i]
            individual_y_train_hat = y_train_hat[:, i]
            individual_y_test = y_test[:, i]
            individual_y_test_hat = y_test_hat[:, i]
            # print("~~~~~~~~~~~~~~~~~~~~~~~")
            # print(y_train.shape)
            # print(y_train_hat.shape)
            # print(y_test.shape)
            # print(y_test_hat.shape)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")

            perf_metrics = self.get_individual_performance_metrics(
                individual_y_train, individual_y_train_hat,
                individual_y_test, individual_y_test_hat, threshold=0.5)

            file_name = "".join(["perf_metrics_cluster", str(i), ".csv"])
            perf_metrics.to_csv(os.path.join(plot_path, file_name))

            plot_name = "".join(["roc", str(i), ".pdf"])
            roc_auc_train = roc_auc_score(individual_y_train, individual_y_train_hat, average="micro")
            fpr_train, tpr_train, _ = roc_curve(individual_y_train, individual_y_train_hat)

            roc_auc_test = roc_auc_score(individual_y_test, individual_y_test_hat, average="micro")
            fpr_test, tpr_test, _ = roc_curve(individual_y_test, individual_y_test_hat)

            plt.figure()
            lw = 2
            plt.plot(fpr_train, tpr_train, color='green',
                     lw=lw, label='ROC Train (AUC = %0.4f)' % roc_auc_train)
            plt.plot(fpr_test, tpr_test, color='darkorange',
                     lw=lw, label='ROC Test (AUC = %0.4f)' % roc_auc_test)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(plot_path,plot_name))


    def get_overall_performance_metrics(self, y_train, y_train_hat, y_test, y_test_hat, threshold=0.5, average="micro"):
        metric_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score']
        metric_values_train = [roc_auc_score(y_train, y_train_hat, average=average),
                               accuracy_score(y_train, y_train_hat > threshold),
                               precision_score(y_train, y_train_hat > threshold, average=average),
                               recall_score(y_train, y_train_hat > threshold, average=average),
                               f1_score(y_train, y_train_hat > threshold, average=average)]
        metric_values_test = [roc_auc_score(y_test, y_test_hat, average=average),
                              accuracy_score(y_test, y_test_hat > threshold),
                              precision_score(y_test, y_test_hat > threshold, average=average),
                              recall_score(y_test, y_test_hat > threshold, average=average),
                              f1_score(y_test, y_test_hat > threshold, average=average)]
        all_metrics = pd.DataFrame({'metrics': metric_names,
                                    'train': metric_values_train,
                                    'test': metric_values_test},
                                   columns=['metrics', 'train', 'test']).set_index('metrics')
        return all_metrics

    def _prepare_micro_roc(self, fpr, tpr, roc_auc, n_class, y, p_y_pred):
        for i in range(n_class):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], p_y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_class):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_class
        return all_fpr, mean_tpr

    def _prepare_overall_roc(self, y_train, y_train_hat, y_test, y_test_hat):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr["micro_train"], tpr["micro_train"], _ = roc_curve(y_train.ravel(), y_train_hat.ravel(), sample_weight=None)
        roc_auc["micro_train"] = auc(fpr["micro_train"], tpr["micro_train"])
        fpr["micro_test"], tpr["micro_test"], _ = roc_curve(y_test.ravel(), y_test_hat.ravel(), sample_weight=None)
        roc_auc["micro_test"] = auc(fpr["micro_test"], tpr["micro_test"])
        fpr["macro_train"], tpr["macro_train"] = self._prepare_micro_roc(fpr, tpr, roc_auc, self.n_class, y_train, y_train_hat)
        roc_auc["macro_train"] = auc(fpr["macro_train"], tpr["macro_train"])
        fpr["macro_test"], tpr["macro_test"] = self._prepare_micro_roc(fpr, tpr, roc_auc, self.n_class, y_test, y_test_hat)
        roc_auc["macro_test"] = auc(fpr["macro_test"], tpr["macro_test"])
        return fpr, tpr, roc_auc

    def plot_overall_roc(self, y_train, y_train_hat, y_test, y_test_hat, plot_path):

        fpr, tpr, roc_auc = self._prepare_overall_roc(y_train, y_train_hat, y_test, y_test_hat)

        plt.figure()
        plt.plot(fpr["macro_train"], tpr["macro_train"],
                 label='macro_train ROC (area = {0:0.2f})'
                       ''.format(roc_auc["macro_test"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro_test"], tpr["macro_test"],
                 label='macro_test ROC (area = {0:0.2f})'
                       ''.format(roc_auc["macro_train"]),
                 color='navy', linestyle=':', linewidth=4)
        plt.plot(fpr["micro_train"], tpr["micro_train"],
                 label='micro_train ROC (area = {0:0.2f})'
                       ''.format(roc_auc["micro_train"]),
                 color='green',
                 linestyle=':', linewidth=4)

        plt.plot(fpr["micro_test"], tpr["micro_test"],
                 label='micro_test ROC (area = {0:0.2f})'
                       ''.format(roc_auc["micro_train"]),
                 color='orange',
                 linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("Receiver operating characteristic to multi-class")
        plt.legend(loc="lower right", prop={'size': 14})
        plt.savefig(plot_path)

    def get_individual_performance_metrics(self, y_train, y_train_hat, y_test, y_test_hat, threshold=0.5):
        metric_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score']
        metric_values_train = [roc_auc_score(y_train, y_train_hat),
                               accuracy_score(y_train, y_train_hat > threshold),
                               precision_score(y_train, y_train_hat > threshold),
                               recall_score(y_train, y_train_hat > threshold),
                               f1_score(y_train, y_train_hat > threshold)]
        metric_values_test = [roc_auc_score(y_test, y_test_hat),
                              accuracy_score(y_test, y_test_hat > threshold),
                              precision_score(y_test, y_test_hat > threshold),
                              recall_score(y_test, y_test_hat > threshold),
                              f1_score(y_test, y_test_hat > threshold)]
        all_metrics = pd.DataFrame({'metrics': metric_names,
                                    'train': metric_values_train,
                                    'test': metric_values_test},
                                   columns=['metrics', 'train', 'test']).set_index('metrics')
        return all_metrics

    def _format(self, y):
        return y[0:y.shape[0], :]