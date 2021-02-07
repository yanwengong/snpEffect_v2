class Evaluator:

    def __init__(self, train_data_loader, test_data_loader, model_path, output_evaluation_data_path):
        self._train_data_loader = train_data_loader
        self._test_data_loader = test_data_loader
        self._mode = torch.load(model_path)
        self._output_evaluation_data_path = output_evaluation_data_path

    def evaluate(self):

        y_train_hat = self._model.predict(self._train_data_loader.get_x())
        y_test_hat = self._model.predict(self._test_data_loader.get_x())

        perf_metrics = self.get_overall_performance_metrics(
            self._train_data_loader.get_y(), self._format(y_train_hat),
            self._test_data_loader.get_y(), self._format(y_test_hat),
            threshold=0.5, average="micro")

        # file_name = "".join(["perf_metrics", ".csv"])
        perf_metrics.to_csv(os.path.join(self._output_evaluation_data_path, "perf_metrics.csv"))

        # plot_name = "".join(["roc", ".pdf"])
        self.plot_overall_roc(
            self._train_data_loader.get_y(), y_train_hat,
            self._test_data_loader.get_y(), y_test_hat,
            os.path.join(self._output_evaluation_data_path, "roc.pdf"))

        self.plot_individual_roc(
            self._train_data_loader.get_y(), y_train_hat,
            self._test_data_loader.get_y(), y_test_hat,
            self._output_evaluation_data_path)

    def plot_individual_roc(self, y_train, y_train_hat, y_test, y_test_hat, plot_path):
        for i in range(y_train.shape[1]):
            individual_y_train = y_train[:, i]
            individual_y_train_hat = y_train_hat[:, i]
            individual_y_test = y_test[:, i]
            individual_y_test_hat = y_test_hat[:, i]
            print("~~~~~~~~~~~~~~~~~~~~~~~")
            print(y_train.shape)
            print(y_train_hat.shape)
            print(y_test.shape)
            print(y_test_hat.shape)
            print("~~~~~~~~~~~~~~~~~~~~~~~~")

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
            plt.savefig(plot_name)

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

    def _prepare_overall_roc(self, y_train, y_train_hat, y_test, y_test_hat):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_class = y_train.shape[1]
        fpr["micro_train"], tpr["micro_train"], _ = roc_curve(y_train.ravel(), y_train_hat.ravel(), sample_weight=None)
        roc_auc["micro_train"] = auc(fpr["micro_train"], tpr["micro_train"])
        fpr["micro_test"], tpr["micro_test"], _ = roc_curve(y_test.ravel(), y_test_hat.ravel(), sample_weight=None)
        roc_auc["micro_test"] = auc(fpr["micro_test"], tpr["micro_test"])
        fpr["macro_train"], tpr["macro_train"] = prepare_micro_roc(fpr, tpr, roc_auc, n_class, y_train, y_train_hat)
        roc_auc["macro_train"] = auc(fpr["macro_train"], tpr["macro_train"])
        fpr["macro_test"], tpr["macro_test"] = prepare_micro_roc(fpr, tpr, roc_auc, n_class, y_test, y_test_hat)
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