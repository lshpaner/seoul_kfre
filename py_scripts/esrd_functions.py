import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    average_precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
)


################################################################################
################################ ESRD Outcome ##################################
################################################################################


def calculate_outcome(df, col, years, duration_col, prefix=None):
    """Calculate outcome based on a given number of years.

    This function creates a new column in the dataframe which is populated with
    a 1 or a 0 based on certain conditions.

    Parameters:
    df (pd.DataFrame): DataFrame to perform calculations on.
    col (str): The column name with ESRD (should be eGFR < 15 flag).
    years (int): The number of years to use in the condition.
    duration_col (str): The name of the column containing the duration data.
    prefix (str): Custom prefix for the new column name. Defaults to "ESRD_in".

    Returns:
    pd.DataFrame: DataFrame with the new column added.
    """
    # Compute the 'years' column if it doesn't already exist
    if "years" not in df.columns:
        df["years"] = round(df[duration_col] / 365.25)

    if prefix is None:
        prefix = "ESRD_in"

    column_name = f"{prefix}_{years}_year_outcome"
    df[column_name] = np.where((df[col] == 1) & (df["years"] <= years), 1, 0)
    return df


################################################################################
######################### CKD Stage Classification #############################
################################################################################


def classify_ckd_stages(
    df,
    egfr_col="eGFR",
    stage_col=None,
    combined_stage_col=None,
):
    """
    Classifies CKD stages based on eGFR values in a specified column of a df.

    Parameters:
    df (pd.DataFrame): DataFrame containing eGFR values.
    egfr_col (str): Name of the column in df containing eGFR values. Default is
                    'eGFR'.
    stage_col (str): Name of the new column to be created for CKD stage.
    combined_stage_col (str): Name of the new column to be created for combined
    CKD stage.

    Returns:
    df (pd.DataFrame): DataFrame with new columns containing CKD stages.
    """

    if stage_col:
        # Define the conditions for each CKD stage according to eGFR values
        conditions = [
            (df[egfr_col] >= 45) & (df[egfr_col] < 60),  # Condition for Stage 3a
            (df[egfr_col] >= 30) & (df[egfr_col] < 45),  # Condition for Stage 3b
            (df[egfr_col] >= 15) & (df[egfr_col] < 30),  # Condition for Stage 4
        ]

        # Define the stage names that correspond to each condition
        choices = ["CKD Stage 3a", "CKD Stage 3b", "CKD Stage 4"]

        # Create a new column in the DataFrame. Use np.select to assign a stage
        # to each row based on the conditions and choices defined above.
        # If none of the conditions are met, the function will return
        # 'Not classified'
        df[stage_col] = np.select(conditions, choices, default="Below Stage 3")

    if combined_stage_col:
        # combine conditions for CKD stages 3 & 4 according to eGFR values
        combined_conditions = [(df[egfr_col] >= 15) & (df[egfr_col] < 60)]

        # Define the stage names that correspond to the combined condition
        combined_choices = ["CKD Stage 3 and 4"]

        # Create a new column in the DataFrame. Use np.select to assign a
        # combined stage to each row based on the conditions and choices defined
        # above. If none of the conditions are met, the function will return
        # 'Not classified'
        df[combined_stage_col] = np.select(
            combined_conditions, combined_choices, default="Not Classified"
        )

    return df


################################################################################
######################## Calculate Performance Metrics #########################
################################################################################


################################################################################


# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    """
    Calculate various evaluation metrics for a binary classification model.

    This function computes a set of performance metrics for a binary classification model
    given the true labels and the predicted probabilities. The metrics calculated include
    precision (positive predictive value), precision-recall AUC, average precision,
    sensitivity (recall), specificity, AUC ROC, and Brier score.

    Parameters:
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).

    y_pred : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns:
    -------
    dict
        A dictionary containing the calculated metrics:
        - "Precision/PPV" (float): Precision or positive predictive value.
        - "PR AUC" (float): Area Under the Precision-Recall Curve.
        - "Average Precision" (float): Average precision score.
        - "Sensitivity" (float): Sensitivity or recall.
        - "Specificity" (float): Specificity (true negative rate).
        - "AUC ROC" (float): Area Under the Receiver Operating Characteristic Curve.
        - "Brier Score" (float): Brier score loss.

    Notes:
    -----
    - Precision is calculated with a threshold of 0.5 for the predicted probabilities.
    - Sensitivity is also known as recall.
    - Specificity is calculated as the recall for the negative class.
    - AUC ROC is calculated using the receiver operating characteristic curve.
    - Brier score measures the mean squared difference between predicted
      probabilities and the true binary outcomes.
    """

    precision = precision_score(y_true, y_pred > 0.5)
    pr_auc = average_precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred > 0.5)
    specificity = recall_score(y_true, y_pred > 0.5, pos_label=0)
    auc_roc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)

    return {
        "Precision/PPV": precision,
        "PR AUC": pr_auc,
        "Average Precision": average_precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "AUC ROC": auc_roc,
        "Brier Score": brier,
    }


# Example usage
outcomes = ["Outcome1", "Outcome2"]
y_true = [np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])]
preds_4var = [np.array([0.1, 0.4, 0.35, 0.8]), np.array([0.05, 0.2, 0.75, 0.9])]

# Initialize an empty list to store the results
metrics_list_4var = []

# Iterate through outcomes and calculate metrics
for outcome, true_labels, pred_labels in zip(outcomes, y_true, preds_4var):
    metrics = calculate_metrics(true_labels, pred_labels)
    metrics["Outcome"] = outcome
    metrics_list_4var.append(metrics)

# Create a DataFrame from the metrics list
metrics_df_4var = pd.DataFrame(metrics_list_4var)

# Transpose the DataFrame for a natural view
metrics_df_4var = metrics_df_4var.set_index("Outcome").T

# Rename the index of the transposed DataFrame
metrics_df_4var = metrics_df_4var.rename_axis("Metrics")

# Display the final DataFrame
print(metrics_df_4var)
