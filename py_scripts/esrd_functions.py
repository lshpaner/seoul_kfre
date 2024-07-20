import pandas as pd
import numpy as np

################################################################################
################################ ESRD Outcome ##################################
################################################################################


def calculate_outcome(df, col, years, prefix=None):
    """Calculate outcome based on a given number of years.

    This function creates a new column in the dataframe which is populated with
    a 1 or a 0 based on certain conditions.

    Parameters:
    df (pd.DataFrame): DataFrame to perform calculations on.
    col (str): The column name with an eGFR < 15 flag.
    years (int): The number of years to use in the condition.
    prefix (str): Custom prefix for the new column name. Defaults to "ESRD_in".

    Returns:
    pd.DataFrame: DataFrame with the new column added.
    """
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
    egfr_col="entryperiod_egfrckdepi2009_mean",
    stage_col=None,
    combined_stage_col=None,
):
    """
    Classifies CKD stages based on eGFR values in a specified column of a df.

    Parameters:
    df (pd.DataFrame): DataFrame containing eGFR values.
    egfr_col (str): Name of the column in df containing eGFR values. Default is
                    'entryperiod_egfrckdepi2009_mean'.
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
