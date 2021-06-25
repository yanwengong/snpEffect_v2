import torch
from torch import nn
import os
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
import pandas as pd
from tqdm import tqdm


class Evaluator:
    def __init__(self, registered_model, train_data, test_data, all_data, model_path, output_evaluation_data_path,
                 batch_size, test_size, n_class):
        '''
        :param train_data: train data
        :param test_data: test_data
        :param model_path: path the model is saved
        :param output_evaluation_data_path: path to save all the evaluation result
        '''
        self.n_class = n_class
        self._mode = registered_model.get_model(self.n_class)
        self._train_data = train_data
        self._test_data = test_data
        self._all_data = all_data
        self._mode.load_state_dict(torch.load(model_path))
        self._output_evaluation_data_path = output_evaluation_data_path
        self._batch_size = batch_size
        self._test_size = test_size


    def evaluate(self):
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("----------Model's state_dict-----------")
        for param_tensor in self._mode.state_dict():
            print(param_tensor, "\t", self._mode.state_dict()[param_tensor].size())

        print("----------weight for multi_head_attention.wq.weight-----------")
        # TODO change this to deepATT only
        # second_query = self._mode.state_dict()["multi_head_attention.wq.weight"]
        # print(second_query.size())
        # print(second_query)
        # second_query = second_query.cpu().detach().numpy()
        # np.savetxt(os.path.join(self._output_evaluation_data_path,"second_stage_query.csv"), second_query, delimiter=",")

        train_loader = torch.utils.data.DataLoader(self._train_data, batch_size=self._batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self._test_data, batch_size=self._batch_size, shuffle=True)

        #train_loader_forward_reverse = torch.utils.data.DataLoader(self._train_data, batch_size=self._batch_size, shuffle=False)
        all_loader = torch.utils.data.DataLoader(self._all_data, batch_size=self._batch_size, shuffle=False)
        test_loader_forward_reverse = torch.utils.data.DataLoader(self._test_data, batch_size=self._batch_size, shuffle=False)


        y_train, p_train_pred = self._predict(train_loader)
        print("--------------_predict y_train output--------------")
        print(type(y_train))
        y_test, p_test_pred = self._predict(test_loader)

        y_all_forward_reverse, p_all_forward_reverse_pred = self._predict(all_loader)
        y_test_forward_reverse, p_test_forward_reverse_pred = self._predict(test_loader_forward_reverse)

        print("-------------- test_forward_reverse shape --------------")
        print(y_test_forward_reverse.shape) #(22908, 8)
        print(p_test_forward_reverse_pred.shape) #(22908, 8)

        ## take max of the predicted forward and reverse

        p_test_fr_pred_final = np.amax((p_test_forward_reverse_pred[0:self._test_size],
                                               p_test_forward_reverse_pred[self._test_size:]), axis=0)
        # p_test_fr_pred_final = np.mean((p_test_forward_reverse_pred[0:self._test_size],
        #                                        p_test_forward_reverse_pred[self._test_size:]), axis=0)
        y_test_fr_final = y_test_forward_reverse[0:self._test_size]
        # double_check = y_test_forward_reverse[self._test_size:]
        # print("-------------- test_forward_reverse double check --------------")
        # print(np.array_equal(double_check, y_test_fr_final))
        # print(double_check)
        # print(y_test_fr_final)

        # self._get_max_indedx(p_train_forward_reverse_pred, "train")
        # self._get_max_indedx(p_test_forward_reverse_pred, "test")
        self._get_max_indedx(p_all_forward_reverse_pred)

        if y_train.shape[1] == p_train_pred.shape[1] == y_test.shape[1] == p_test_pred.shape[1] == 1:
            print("--------------binary prediction--------------")
            self._plot_roc_curve(y_train, p_train_pred, y_test, p_test_pred, self._output_evaluation_data_path, "binary")
            self._get_performance_metrics(y_train, p_train_pred, y_test, p_test_pred, self._output_evaluation_data_path, "binary")
            print("binary metrics and roc curve done")

        else:
            print("--------------multi-label classification--------------")
            print(y_train.shape[1])

            # generate individual metrics
            for i in range(y_train.shape[1]):
                y_train_i = y_train[:, i]
                p_train_pred_i = p_train_pred[:, i]
                y_test_i = y_test[:, i]
                p_test_pred_i = p_test_pred[:, i]
                print("------individual metrics shape------")
                print(y_train_i.shape)
                print(p_train_pred_i.shape)
                print(y_test_i.shape)
                print(p_test_pred_i.shape)
                self._plot_roc_curve(y_train_i, p_train_pred_i, y_test_i, p_test_pred_i, self._output_evaluation_data_path, "cell"+str(i))
                self._get_performance_metrics(y_train_i, p_train_pred_i, y_test_i, p_test_pred_i,
                                              self._output_evaluation_data_path, "cell"+str(i))
            print("individual metrics and roc curve done")

            # generate overall metrics
            print("--------------_get_overall_performance_metrics y_train input--------------")
            print(type(y_train))
            self._get_overall_performance_metrics(y_train, p_train_pred, y_test, p_test_pred, self._output_evaluation_data_path)
            self._plot_overall_roc(y_train, p_train_pred, y_test, p_test_pred, self._output_evaluation_data_path)
            self._get_overall_performance_metrics_fr(y_test_fr_final, p_test_fr_pred_final, self._output_evaluation_data_path)

            print("multi-label metrics and roc curve done")



    def _predict(self, data_loader):
        self._mode.eval()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._mode.to(device)
        print(self._mode.type)
        with torch.no_grad():

            y_arr = np.empty((0, self.n_class))
            y_hat_arr = np.empty((0, self.n_class))
            test_loss = 0.0

            for X, y in tqdm(data_loader):
                X = X.to(device)
                y = y.to(device)
                outputs = self._mode(X.float())

                y_hat = outputs
                y = y.float()

                #test_loss += nn.BCELoss()(y_hat, y).item() * X.size(0)  # sum up batch loss
                test_loss += nn.BCEWithLogitsLoss()(y_hat, y).item() * X.size(0)
                y_arr = np.concatenate((y_arr, y.cpu().numpy()))
                y_hat_arr = np.concatenate((y_hat_arr, y_hat.cpu().numpy()))

            test_loss = test_loss/len(data_loader.dataset)
            print("------------------ data shape --------------------")
            print(y_arr.shape)
            print(y_hat_arr.shape)
            print(type(y_arr)) #<class 'numpy.ndarray'>
            print(type(y_hat_arr)) #<class 'numpy.ndarray'>
            # print(y_arr)
            # print(y_hat_arr)
            y_hat_arr = 1 / (1 + np.exp(-y_hat_arr))  # 0320 evening, add sigmoid
            # print(y_hat_arr)

            # pd.DataFrame(y_hat_arr).to_csv(os.path.join(self._output_evaluation_data_path,
            #                                             file_name+"_y_hat.csv"))
            print(np.count_nonzero(y_arr))
            print('Test loss {}'.format(test_loss))
            print("------------------ train/test prediction done --------------------")

            return y_arr, y_hat_arr


    def _get_max_indedx(self, p_pred):

        """
        function to get the index of max and reverse
        :param p_pred: the predicted score, first half is forward, second half is reverse
        #:param type: "train" or "test"
        return: in each cluster, return two np array containing the index should be included as forward/reverse
        """

        size = int(p_pred.shape[0]/2)
        max_index = np.argmax((p_pred[0:size],
                               p_pred[size:]), axis=0)
        for i in range(p_pred.shape[1]):
            cluster_forward_index = np.argwhere(max_index[:, i] == 0)
            cluster_reverse_index = np.argwhere(max_index[:, i] == 1)

            forward_file_name = "forward_cluster_" + str(i) + "_" + "index.npy"
            np.save(os.path.join(self._output_evaluation_data_path, forward_file_name),
                    cluster_forward_index)

            reverse_file_name = "reverse_cluster_" + str(i) + "_" + "index.npy"
            np.save(os.path.join(self._output_evaluation_data_path, reverse_file_name),
                    cluster_reverse_index)


    def _format(self, y):
        return y[0:y.shape[0], :]

    def _plot_roc_curve(self, y_train, p_train_pred, y_test, p_test_pred, plot_path, cell):
        roc_auc_train = roc_auc_score(y_train, p_train_pred)
        fpr_train, tpr_train, _ = roc_curve(y_train, p_train_pred)

        roc_auc_test = roc_auc_score(y_test, p_test_pred)
        fpr_test, tpr_test, _ = roc_curve(y_test, p_test_pred)
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
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plot_name = cell + "_" + "roc.pdf"
        plt.savefig(os.path.join(plot_path, plot_name))

    def _get_performance_metrics(self, y_train, p_train_pred, y_test, p_test_pred, output_path, cell, threshold=0.5):
        metric_names = ['AUC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'f1-score']
        metric_values_train = [roc_auc_score(y_train, p_train_pred),
                               average_precision_score(y_train, p_train_pred),
                               accuracy_score(y_train, p_train_pred > threshold),
                               precision_score(y_train, p_train_pred > threshold),
                               recall_score(y_train, p_train_pred > threshold),
                               f1_score(y_train, p_train_pred > threshold)
                               ]
        metric_values_train = np.array(metric_values_train).round(3)
        metric_values_test = [roc_auc_score(y_test, p_test_pred),
                              average_precision_score(y_test, p_test_pred),
                              accuracy_score(y_test, p_test_pred > threshold),
                              precision_score(y_test, p_test_pred > threshold),
                              recall_score(y_test, p_test_pred > threshold),
                              f1_score(y_test, p_test_pred > threshold)
                              ]
        metric_values_test = np.array(metric_values_test).round(3)
        all_metrics = pd.DataFrame({'metrics': metric_names,
                                    'train': metric_values_train,
                                    'test': metric_values_test}, columns=['metrics', 'train', 'test']).set_index(
            'metrics')
        file_name = cell + "_" + "perf_metrics.csv"
        all_metrics.to_csv(os.path.join(output_path, file_name))



        ################## before 02/26 ###############################
        # # calculate and save overall performance metrics
        # perf_metrics = self.get_overall_performance_metrics(
        #     y_train, self._format(y_train_hat),
        #     y_test, self._format(y_test_hat),
        #     threshold=0.5, average="micro")
        #
        # perf_metrics.to_csv(os.path.join(self._output_evaluation_data_path, "perf_metrics.csv"))
        #
        # print("overall metrics calculation done")
        #
        # # plot overall roc
        # self.plot_overall_roc(
        #     y_train, y_train_hat,
        #     y_test, y_test_hat,
        #     os.path.join(self._output_evaluation_data_path, "roc.pdf"))
        #
        # print("overall roc curve done")
        #
        # # ploc overall roc and save individual metrics
        # print("----------below is y_train_hat---------")
        # print(y_train_hat)
        # print("----------below is y_test_hat----------")
        # print(y_test_hat)
        # self.plot_individual_roc(
        #     y_train, y_train_hat,
        #     y_test, y_test_hat,
        #     self._output_evaluation_data_path)
        #
        # print("individual metrics and roc curve done")
        ################## before 02/26 ###############################

    # def _plot_individual_roc(self, y_train, y_train_hat, y_test, y_test_hat, plot_path):
    #     for i in range(y_train.shape[1]):
    #         individual_y_train = y_train[:, i]
    #         individual_y_train_hat = y_train_hat[:, i]
    #         individual_y_test = y_test[:, i]
    #         individual_y_test_hat = y_test_hat[:, i]
    #
    #         perf_metrics = self.get_individual_performance_metrics(
    #             individual_y_train, individual_y_train_hat,
    #             individual_y_test, individual_y_test_hat, threshold=0.5)
    #
    #         file_name = "".join(["perf_metrics_cluster", str(i), ".csv"])
    #         perf_metrics.to_csv(os.path.join(plot_path, file_name))
    #
    #         plot_name = "".join(["roc", str(i), ".pdf"])
    #         roc_auc_train = roc_auc_score(individual_y_train, individual_y_train_hat, average="micro")
    #         fpr_train, tpr_train, _ = roc_curve(individual_y_train, individual_y_train_hat)
    #
    #         roc_auc_test = roc_auc_score(individual_y_test, individual_y_test_hat, average="micro")
    #         fpr_test, tpr_test, _ = roc_curve(individual_y_test, individual_y_test_hat)
    #
    #         plt.figure()
    #         lw = 2
    #         plt.plot(fpr_train, tpr_train, color='green',
    #                  lw=lw, label='ROC Train (AUC = %0.4f)' % roc_auc_train)
    #         plt.plot(fpr_test, tpr_test, color='darkorange',
    #                  lw=lw, label='ROC Test (AUC = %0.4f)' % roc_auc_test)
    #         plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #         plt.xlim([0.0, 1.0])
    #         plt.ylim([0.0, 1.05])
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.title('Receiver operating characteristic curve')
    #         plt.legend(loc="lower right")
    #         plt.savefig(os.path.join(plot_path,plot_name))

    def _calculate_micro_acc(self, y, y_hat):
        y_pred = y_hat.round() # convert prob to 0/1
        correct = np.sum(y == y_pred)
        acc = float(correct) / (y.shape[0]*y.shape[1])
        return acc


    def _get_overall_performance_metrics(self, y_train, y_train_hat, y_test, y_test_hat, output_path, threshold=0.5, average="micro"):
        metric_names = ['AUC', 'AUPR', 'Exact_match_ratio', 'Precision', 'Recall', 'f1-score']
        print("----------y_train----------")
        print(y_train)
        print("----------y_train_hat----------")
        print(y_train_hat)
        metric_values_train = [roc_auc_score(y_train, y_train_hat, average=average),
                               average_precision_score(y_train, y_train_hat, average=average),
                               self._calculate_micro_acc(y_train, y_train_hat), # the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
                               precision_score(y_train, y_train_hat > threshold, average=average),
                               recall_score(y_train, y_train_hat > threshold, average=average),
                               f1_score(y_train, y_train_hat > threshold, average=average)]
        metric_values_train = np.array(metric_values_train).round(3)
        metric_values_test = [roc_auc_score(y_test, y_test_hat, average=average),
                              average_precision_score(y_test, y_test_hat, average=average),
                              self._calculate_micro_acc(y_test, y_test_hat),
                              precision_score(y_test, y_test_hat > threshold, average=average),
                              recall_score(y_test, y_test_hat > threshold, average=average),
                              f1_score(y_test, y_test_hat > threshold, average=average)]
        metric_values_test = np.array(metric_values_test).round(3)
        all_metrics = pd.DataFrame({'metrics': metric_names,
                                    'train': metric_values_train,
                                    'test': metric_values_test},
                                   columns=['metrics', 'train', 'test']).set_index('metrics')
        file_name = "overall_perf_metrics.csv"
        all_metrics.to_csv(os.path.join(output_path, file_name))



    def _get_overall_performance_metrics_fr(self, y_train, y_train_hat, output_path, threshold=0.5, average="micro"):
        metric_names = ['AUC', 'AUPR', 'Exact_match_ratio', 'Precision', 'Recall', 'f1-score']
        print("----------y_train----------")
        print(y_train)
        print("----------y_train_hat----------")
        print(y_train_hat)
        metric_values_train = [roc_auc_score(y_train, y_train_hat, average=average),
                               average_precision_score(y_train, y_train_hat, average=average),
                               self._calculate_micro_acc(y_train, y_train_hat), # the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
                               precision_score(y_train, y_train_hat > threshold, average=average),
                               recall_score(y_train, y_train_hat > threshold, average=average),
                               f1_score(y_train, y_train_hat > threshold, average=average)]
        metric_values_train = np.array(metric_values_train).round(3)

        all_metrics = pd.DataFrame({'metrics': metric_names,
                                    'train': metric_values_train},
                                   columns=['metrics', 'train', 'test']).set_index('metrics')
        file_name = "overall_forward_reverse_perf_metrics.csv"
        all_metrics.to_csv(os.path.join(output_path, file_name))

    def _prepare_macro_roc(self, fpr, tpr, roc_auc, n_class, y, p_y_pred):
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
        fpr["macro_train"], tpr["macro_train"] = self._prepare_macro_roc(fpr, tpr, roc_auc, self.n_class, y_train, y_train_hat)
        roc_auc["macro_train"] = auc(fpr["macro_train"], tpr["macro_train"])
        fpr["macro_test"], tpr["macro_test"] = self._prepare_macro_roc(fpr, tpr, roc_auc, self.n_class, y_test, y_test_hat)
        roc_auc["macro_test"] = auc(fpr["macro_test"], tpr["macro_test"])
        return fpr, tpr, roc_auc

    def _plot_overall_roc(self, y_train, y_train_hat, y_test, y_test_hat, plot_path):

        fpr, tpr, roc_auc = self._prepare_overall_roc(y_train, y_train_hat, y_test, y_test_hat)

        plt.figure()
        plt.plot(fpr["macro_train"], tpr["macro_train"],
                 label='macro_train ROC (area = {0:0.2f})'
                       ''.format(roc_auc["macro_train"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro_test"], tpr["macro_test"],
                 label='macro_test ROC (area = {0:0.2f})'
                       ''.format(roc_auc["macro_test"]),
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
        plot_name = "roc.pdf"
        plt.savefig(os.path.join(plot_path, plot_name))


    # def get_individual_performance_metrics(self, y_train, y_train_hat, y_test, y_test_hat, threshold=0.5):
    #     metric_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score']
    #     metric_values_train = [roc_auc_score(y_train, y_train_hat),
    #                            accuracy_score(y_train, y_train_hat > threshold),
    #                            precision_score(y_train, y_train_hat > threshold),
    #                            recall_score(y_train, y_train_hat > threshold),
    #                            f1_score(y_train, y_train_hat > threshold)]
    #     metric_values_train = np.array(metric_values_train).round(3)
    #     metric_values_test = [roc_auc_score(y_test, y_test_hat),
    #                           accuracy_score(y_test, y_test_hat > threshold),
    #                           precision_score(y_test, y_test_hat > threshold),
    #                           recall_score(y_test, y_test_hat > threshold),
    #                           f1_score(y_test, y_test_hat > threshold)]
    #     metric_values_test = np.array(metric_values_test).round(3)
    #     all_metrics = pd.DataFrame({'metrics': metric_names,
    #                                 'train': metric_values_train,
    #                                 'test': metric_values_test},
    #                                columns=['metrics', 'train', 'test']).set_index('metrics')
    #     return all_metrics
    #




    # def _predict2(self, data_loader):
    #     self._mode.eval()
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     test_loss = 0
    #     correct = 0
    #     y_pred = []
    #     y_true = []
    #     y_proba = []
    #
    #     with torch.no_grad():
    #         for data, target in tqdm(data_loader):
    #             data, target = data.to(device), target.to(device)
    #             output = self._mode(data).squeeze()
    #             test_loss += nn.BCELoss()(output, target).item() * 1e+1  # sum up batch loss
    #             prob = (output)
    #             y_probas = prob.cpu().numpy()
    #             prob[prob >= 0.5] = 1
    #             prob[prob < 0.5] = 0
    #             correct += torch.sum(prob == target)
    #             for i in range(len(prob)):
    #                 y_proba.append(float(y_probas[i]))
    #                 y_pred.append(float(prob[i]))
    #                 y_true.append(float(target[i]))
    #         #target_names = ['Negative', 'Positive']
    #         # roc(y_true,y_proba)
    #         # prc(y_true,y_proba)
    #         #print(classification_report(y_true, y_pred, target_names=target_names))
    #
    #         print("---------new y_proba--------------")
    #         print(y_proba)
    #         print("---------new y_proba done--------------")
    #         return y_pred, y_true, y_proba

