from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from torch.utils.data import Subset

from src.dataset import TabularDataset
from src.utils import remove_subsets


class Massaging:
    def __init__(self, sensitive_group):
        self.sensitive_group = sensitive_group

    def fit(self, X):
        X = remove_subsets(X)
        train_data = pd.DataFrame(X.features, columns=X.feature_names)

        sensitive_attribute = X.protected_attributes.ravel()
        labels = X.labels.ravel()

        protected_indices = np.where(sensitive_attribute == self.sensitive_group)[0]
        unprotected_indices = np.where(sensitive_attribute != self.sensitive_group)[0]

        pos_decision_ratio_protected_indices = sum(labels[protected_indices]) / max(len(
            protected_indices
        ), 1)
        pos_decision_ratio_unprotected_indices = sum(labels[unprotected_indices]) / max(len(
            unprotected_indices
        ), 1)

        disc = (
            pos_decision_ratio_unprotected_indices
            - pos_decision_ratio_protected_indices
        )
        number_of_protected = len(protected_indices)
        number_of_unprotected = len(unprotected_indices)
        total_number_of_instances = len(train_data)

        # number of promotor, demotor pairs to achieve demographic parity
        M = round(
            (disc * number_of_protected * number_of_unprotected)
            / total_number_of_instances
        )
        predicted_probabilities = self.learn_classifier(train_data, labels)

        predicted_probabilities_unprotected = pd.DataFrame(
            predicted_probabilities[unprotected_indices]
        )
        predicted_probabilities_unprotected["Index"] = unprotected_indices

        predicted_probabilities_protected = pd.DataFrame(
            predicted_probabilities[protected_indices]
        )
        predicted_probabilities_protected["Index"] = protected_indices

        if M > 0:
            self.demotion_candidates = self.get_doubtful_positive_cases(
                predicted_probabilities_unprotected, labels[unprotected_indices], M
            )["Index"]
            self.promotion_candidates = self.get_doubtful_negative_cases(
                predicted_probabilities_protected, labels[protected_indices], M
            )["Index"]
        else:
            self.promotion_candidates = []
            self.demotion_candidates = []

        return

    def transform(self, X):
        if isinstance(X, Subset):
            transformed_data = remove_subsets(X).copy()
        else:
            transformed_data = X.copy(deepcopy=True)
        new_class_labels = transformed_data.labels.copy()

        np.put(new_class_labels, self.demotion_candidates, 0)
        np.put(new_class_labels, self.promotion_candidates, 1)

        if isinstance(transformed_data, TabularDataset):
            transformed_data.set_labels(new_class_labels)
            return transformed_data
        else:
            transformed_data.labels = new_class_labels
            return transformed_data

    def learn_classifier(self, X, y):
        LR = LogisticRegression()
        LR.fit(X, y)
        predictions_probabilities = LR.predict_proba(X)
        return predictions_probabilities

    def order_instances(self, probability_labels):
        sort_by_probability = probability_labels.sort_values(1)
        sort_by_probability = sort_by_probability.reset_index(drop=True)
        return sort_by_probability

    # these are the cases of the unprotected indices that need to change
    def get_doubtful_positive_cases(self, probability_labels, class_labels, M):
        indices_with_positive_class_label = np.where(class_labels == 1)[0]
        probability_labels_of_positive_class_labels = probability_labels.iloc[
            indices_with_positive_class_label
        ]
        sorted_probability_labels = self.order_instances(
            probability_labels_of_positive_class_labels
        )
        demotion_candidates = sorted_probability_labels.iloc[0:M]
        return demotion_candidates

    # these are the cases of the protected indices that need to change
    def get_doubtful_negative_cases(self, probability_labels, class_labels, M):
        indices_with_negative_class_label = np.where(class_labels == 0)[0]
        probability_labels_of_negative_class_labels = probability_labels.iloc[
            indices_with_negative_class_label
        ]
        sorted_probability_labels = self.order_instances(
            probability_labels_of_negative_class_labels
        )
        promotion_candidates = sorted_probability_labels.iloc[-M:]
        return promotion_candidates

    def get_demotion_candidates(self):
        return self.demotion_candidates

    def get_promotion_candidates(self):
        return self.promotion_candidates
