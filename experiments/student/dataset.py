from pathlib import Path

import pandas as pd
from aif360.datasets import StandardDataset

DATA_ROOT = Path("res/data/student")

# First function to be called to load the collected dataset, with its biased and unbiased grades
def load_data_with_biased_and_unbiased_grades():
    raw_data_with_predictions = pd.read_csv(
        DATA_ROOT / Path("CompleteDataAndBiases.csv"), index_col="index"
    )

    categorical_attributes = ["romantic", "reason"]
    return raw_data_with_predictions, categorical_attributes
   # return add_columns_from_org_data(["school","address","famsize","Pstatus","Mjob","Fjob","guardian","schoolsup","famsup","paid","activities","nursery","higher","internet","subject",],
   #                                   raw_data_with_predictions,
   #                                   categorical_attributes)


# If desired some columns from the original dataset can be added to the dataset containing the fair and biased grades
def add_columns_from_org_data(
    columns_to_add, data_with_biased_and_fair_grades, categorical_attributes
):
    categorical_attributes_org_data = {
        "school",
        "address",
        "famsize",
        "Pstatus",
        "Mjob",
        "Fjob",
        "guardian",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "subject",
    }
    raw_original_data = pd.read_csv(DATA_ROOT / Path("original_data.csv"))
    original_data_columns_to_add = raw_original_data[columns_to_add].loc[
        data_with_biased_and_fair_grades.index
    ]
    data_with_biased_and_fair_grades[columns_to_add] = original_data_columns_to_add

    categorical_attributes_added_to_data = categorical_attributes_org_data.intersection(
        set(columns_to_add)
    )
    categorical_attributes.extend(list(categorical_attributes_added_to_data))

    return data_with_biased_and_fair_grades, categorical_attributes


# Preprocessing function: Here the raw data will be seperated into two dataframes. One contains all the information of the fair labels, while the other one contains
# the same information with the biased grades. Which version of the biased grades is taken, dependens on the input for variable "binary_labels_conversion_strategy"
# (either "Ranking" or "Pass-Fail").
def preprocess_data(
    data,
    categorical_attributes,
    one_hot_encoding=False,
    binary_labels_conversion_strategy="Ranking",
):
    """
    :param data: pandas Dataframe - Data containing the biased and unbiased version of the grades
    :param one_hot_encoding: boolean - Denotes whether one-hot-encoding should be part of the preprocessing or not
    :param categorical_attributes: list of string - List of all the categorical attribute names that will be one-hot-encoded if specified
    :param binary_labels_conversion_strategy: string - Specifies which version of the biased decision labels to use
    :return: pandas dataframe, pandas datframe - dataframe containing the fair version of the labels, dataframe containing the biased version of the labelste
    """
    if binary_labels_conversion_strategy == "Pass-Fail":
        biased_label_column = "Predicted_Pass_PassFailStrategy"
        other_biased_label_column = "Predicted_Pass_RankingStrategy"
    else:
        biased_label_column = "Predicted_Pass_RankingStrategy"
        other_biased_label_column = "Predicted_Pass_PassFailStrategy"

    data = data.drop(
        [
            "ParticipantID",
            "name",
            "G3",
            "PredictedRank",
            "PredictedGrade",
            "StereotypeActivation",
        ],
        axis=1,
    )  # "StereotypeActivation"

    if one_hot_encoding:
        one_hot = pd.get_dummies(data[categorical_attributes])
        data = data.drop(categorical_attributes, axis=1)
        data = data.join(one_hot)
        categorical_attributes = []

    data['sex'] = data['sex'].map({'F': 0, 'M': 1}) # if not StandardDataset maps privileged class to 1 (we use A = 0 as privileged/non-sensitive)

    for label_column in ["Pass", biased_label_column, other_biased_label_column]:
        data[label_column] = data[label_column].map({True: 1, False: 0})

    fair_data = data.copy()
    fair_data = fair_data.drop([biased_label_column, other_biased_label_column], axis=1)

    biased_data = data.drop(["Pass", other_biased_label_column], axis=1)
    biased_data = biased_data.rename(columns={biased_label_column: "Pass"})

    return StandardDataset(
        fair_data,
        label_name='Pass',
        favorable_classes=[1],
        protected_attribute_names=["sex"],
        privileged_classes=[[0]],
        categorical_features=categorical_attributes,
    ), StandardDataset(
        biased_data,
        label_name='Pass',
        favorable_classes=[1],
        protected_attribute_names=["sex"],
        privileged_classes=[[0]],
        categorical_features=categorical_attributes,
    )
