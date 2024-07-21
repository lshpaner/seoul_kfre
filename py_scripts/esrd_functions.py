import pandas as pd
import numpy as np
import os
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
    Classifies CKD stages based on eGFR values in a specified column of a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing eGFR values.
    egfr_col (str): Name of the column in df containing eGFR values. Default is 'eGFR'.
    stage_col (str): Name of the new column to be created for CKD stage.
    combined_stage_col (str): Name of the new column to be created for combined CKD stage.

    Returns:
    pd.DataFrame: DataFrame with new columns containing CKD stages.
    """

    if stage_col:
        # Define the conditions for each CKD stage according to eGFR values
        conditions = [
            (df[egfr_col] >= 90),  # Condition for Stage 1
            (df[egfr_col] >= 60) & (df[egfr_col] < 90),  # Condition for Stage 2
            (df[egfr_col] >= 45) & (df[egfr_col] < 60),  # Condition for Stage 3a
            (df[egfr_col] >= 30) & (df[egfr_col] < 45),  # Condition for Stage 3b
            (df[egfr_col] >= 15) & (df[egfr_col] < 30),  # Condition for Stage 4
            (df[egfr_col] < 15),  # Condition for Stage 5
        ]

        # Define the stage names that correspond to each condition
        choices = [
            "CKD Stage 1",
            "CKD Stage 2",
            "CKD Stage 3a",
            "CKD Stage 3b",
            "CKD Stage 4",
            "CKD Stage 5",
        ]

        # Create a new column in the DataFrame
        df[stage_col] = np.select(conditions, choices, default="Not classified")

    if combined_stage_col:
        # Combine conditions for CKD stages 3, 4, and 5 according to eGFR values
        combined_conditions = df[egfr_col] < 60

        # Define the stage names that correspond to the combined condition
        combined_choices = ["CKD Stage 3 - 5"]

        # Create a new column in the DataFrame
        df[combined_stage_col] = np.select(
            [combined_conditions], combined_choices, default="Not classified"
        )

    return df


################################################################################
######################## Calculate Performance Metrics #########################
################################################################################


def prep_and_plot_metrics_vars(
    df,
    num_vars,
    fig_size=(12, 6),
    mode="both",
    image_path_png=None,
    image_path_svg=None,
    image_prefix=None,
    bbox_inches="tight",
    plot_type="both",
    save_plots=True,
    show_years=[2, 5],
    plot_combinations=False,
    show_grids=False,
):
    """
    Generate the true labels and predicted probabilities for 2-year and 5-year outcomes,
    and optionally plot and save ROC and Precision-Recall curves for specified variable models.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the necessary columns for truth and predictions.

    num_vars : int, list of int, or tuple of int
        Number of variables (e.g., 4) or a list/tuple of numbers of variables
        (e.g., [4, 6]) to generate predictions for.

    fig_size : tuple of int, optional
        Size of the figure for the ROC plot, default is (12, 6).

    mode : str, optional
        Operation mode, can be 'prep', 'plot', or 'both'. Default is 'both'.
        'prep' only prepares the metrics,
        'plot' only plots the metrics (requires pre-prepped metrics),
        'both' prepares and plots the metrics.

    image_path_png : str, optional
        Path to save the PNG images. Default is None.

    image_path_svg : str, optional
        Path to save the SVG images. Default is None.

    image_prefix : str, optional
        Prefix to use for saved images. Default is None.

    bbox_inches : str, optional
        Bounding box in inches for the saved images. Default is 'tight'.

    plot_type : str, optional
        Type of plot to generate, can be 'roc', 'pr', or 'both'. Default is 'both'.

    save_plots : bool, optional
        Whether to save plots. Default is True.

    show_years : int, list of int, or tuple of int, optional
        Year outcomes to show in the plots. Default is [2, 5].

    plot_combinations : bool, optional
        Whether to plot all combinations of variables in a single plot. Default is False.

    show_grids : bool, optional
        Whether to show grid plots of all combinations. Default is False.

    Returns:
    -------
    tuple (optional)
        Only returned if mode is 'prep' or 'both':
        - y_true (list of pd.Series): True labels for specified year outcomes.
        - preds (dict of list of pd.Series): Predicted probabilities for each
          number of variables and each outcome.
        - outcomes (list of str): List of outcome labels.

    Raises:
    -------
    ValueError
        If 'save_plots' is True without specifying 'image_path_png' or 'image_path_svg'.
        If 'bbox_inches' is not a string or None.
        If 'show_years' contains invalid year values.
    """
    valid_years = [2, 5]
    if isinstance(show_years, int):
        show_years = [show_years]
    elif isinstance(show_years, tuple):
        show_years = list(show_years)

    if any(year not in valid_years for year in show_years):
        raise ValueError(
            f"The 'show_years' parameter must be a list or tuple containing any of {valid_years}."
        )

    # Ensure num_vars is a list even if a single int is provided
    if isinstance(num_vars, int):
        num_vars = [num_vars]
    elif isinstance(num_vars, tuple):
        num_vars = list(num_vars)

    # Check for invalid image saving configuration
    if save_plots and not (image_path_png or image_path_svg):
        raise ValueError(
            "To save plots, 'image_path_png' or 'image_path_svg' must be specified."
        )

    # Ensure bbox_inches is a string or None
    if not isinstance(bbox_inches, (str, type(None))):
        raise ValueError("The 'bbox_inches' parameter must be a string or None.")

    y_true = []
    outcomes = []
    for year in show_years:
        y_true.append(df[f"{year}_year_outcome"])
        outcomes.append(f"{year}-year")

    # Prepare predictions
    preds = {}
    for n in num_vars:
        preds[f"{n}var"] = [df[f"kfre_{n}var_{year}year"] for year in show_years]

    if mode in ["prep", "both"]:
        result = (y_true, preds, outcomes)
        if mode == "prep":
            return result

    roc_figs, pr_figs = [], []

    if mode in ["plot", "both"]:
        if plot_combinations:
            # Plot all variable combinations in a single plot for each year outcome
            if plot_type in ["roc", "both"]:
                fig = plt.figure(figsize=fig_size)
                for n in num_vars:
                    for true_labels, pred_labels, outcome in zip(
                        y_true, preds[f"{n}var"], outcomes
                    ):
                        fpr, tpr, _ = roc_curve(true_labels, pred_labels)
                        auc_score = auc(fpr, tpr)
                        plt.plot(
                            fpr,
                            tpr,
                            label=f"{n}-variable {outcome} outcome (AUC = {auc_score:.02f})",
                        )
                plt.plot([0, 1], [0, 1], linestyle="--", color="red")
                plt.xlabel("1 - Specificity")
                plt.ylabel("Sensitivity")
                plt.title(f"AUC ROC for KFRE Outcomes with Different Variables")
                plt.legend(loc="best")
                if save_plots and not show_grids:
                    filename = (
                        f"{image_prefix}_roc_curve_combined"
                        if image_prefix
                        else "roc_curve_combined"
                    )
                    if image_path_png:
                        os.makedirs(image_path_png, exist_ok=True)
                        plt.savefig(
                            os.path.join(image_path_png, f"{filename}.png"),
                            bbox_inches=bbox_inches,
                        )
                    if image_path_svg:
                        os.makedirs(image_path_svg, exist_ok=True)
                        plt.savefig(
                            os.path.join(image_path_svg, f"{filename}.svg"),
                            bbox_inches=bbox_inches,
                        )
                roc_figs.append(fig)
                if not show_grids:
                    plt.show()
                else:
                    plt.close(fig)

            if plot_type in ["pr", "both"]:
                fig = plt.figure(figsize=fig_size)
                for n in num_vars:
                    for true_labels, pred_labels, outcome in zip(
                        y_true, preds[f"{n}var"], outcomes
                    ):
                        precision, recall, _ = precision_recall_curve(
                            true_labels, pred_labels
                        )
                        ap_score = average_precision_score(true_labels, pred_labels)
                        plt.plot(
                            recall,
                            precision,
                            label=f"{n}-variable {outcome} outcome (AP = {ap_score:.02f})",
                        )
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(
                    f"Precision-Recall Curve for Outcomes with Different Variables"
                )
                plt.legend(loc="best")
                if save_plots and not show_grids:
                    filename = (
                        f"{image_prefix}_pr_curve_combined"
                        if image_prefix
                        else "pr_curve_combined"
                    )
                    if image_path_png:
                        os.makedirs(image_path_png, exist_ok=True)
                        plt.savefig(
                            os.path.join(image_path_png, f"{filename}.png"),
                            bbox_inches=bbox_inches,
                        )
                    if image_path_svg:
                        os.makedirs(image_path_svg, exist_ok=True)
                        plt.savefig(
                            os.path.join(image_path_svg, f"{filename}.svg"),
                            bbox_inches=bbox_inches,
                        )
                pr_figs.append(fig)
                if not show_grids:
                    plt.show()
                else:
                    plt.close(fig)
        else:
            # Plot each variable and year combination in separate plots
            for n in num_vars:
                pred_list = preds[f"{n}var"]
                if plot_type in ["roc", "both"]:
                    fig = plt.figure(figsize=fig_size)
                    for i, (true_labels, outcome) in enumerate(zip(y_true, outcomes)):
                        pred_labels = pred_list[i]
                        fpr, tpr, _ = roc_curve(true_labels, pred_labels)
                        auc_score = auc(fpr, tpr)
                        plt.plot(
                            fpr,
                            tpr,
                            label=f"{n}-variable {outcome} outcome (AUC = {auc_score:.02f})",
                        )
                    plt.plot(
                        [0, 1], [0, 1], linestyle="--", color="red"
                    )  # Ensure diagonal line is dotted and red
                    plt.xlabel("1 - Specificity")
                    plt.ylabel("Sensitivity")
                    plt.title(f"AUC ROC for Outcomes with {n} Variables")
                    plt.legend(loc="best")
                    if save_plots and not show_grids:
                        filename = (
                            f"{image_prefix}_{n}var_roc_curve"
                            if image_prefix
                            else f"{n}var_roc_curve"
                        )
                        if image_path_png:
                            os.makedirs(image_path_png, exist_ok=True)
                            plt.savefig(
                                os.path.join(image_path_png, f"{filename}.png"),
                                bbox_inches=bbox_inches,
                            )
                        if image_path_svg:
                            os.makedirs(image_path_svg, exist_ok=True)
                            plt.savefig(
                                os.path.join(image_path_svg, f"{filename}.svg"),
                                bbox_inches=bbox_inches,
                            )
                    roc_figs.append(fig)
                    if not show_grids:
                        plt.show()
                    else:
                        plt.close(fig)

                if plot_type in ["pr", "both"]:
                    fig = plt.figure(figsize=fig_size)
                    for i, (true_labels, outcome) in enumerate(zip(y_true, outcomes)):
                        pred_labels = pred_list[i]
                        precision, recall, _ = precision_recall_curve(
                            true_labels, pred_labels
                        )
                        ap_score = average_precision_score(true_labels, pred_labels)
                        plt.plot(
                            recall,
                            precision,
                            label=f"{n}-variable {outcome} outcome (AP = {ap_score:.02f})",
                        )
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title(f"Precision-Recall Curve for Outcomes with {n} Variables")
                    plt.legend(loc="best")
                    if save_plots and not show_grids:
                        filename = (
                            f"{image_prefix}_{n}var_pr_curve"
                            if image_prefix
                            else f"{n}var_pr_curve"
                        )
                        if image_path_png:
                            os.makedirs(image_path_png, exist_ok=True)
                            plt.savefig(
                                os.path.join(image_path_png, f"{filename}.png"),
                                bbox_inches=bbox_inches,
                            )
                        if image_path_svg:
                            os.makedirs(image_path_svg, exist_ok=True)
                            plt.savefig(
                                os.path.join(image_path_svg, f"{filename}.svg"),
                                bbox_inches=bbox_inches,
                            )
                    pr_figs.append(fig)
                    if not show_grids:
                        plt.show()
                    else:
                        plt.close(fig)

        # Create and save grid plots if show_grids is True
        if show_grids:
            grid_figs = roc_figs + pr_figs
            if grid_figs:
                grid_cols = min(len(grid_figs), 3)
                grid_rows = (len(grid_figs) + grid_cols - 1) // grid_cols
                fig, axs = plt.subplots(
                    grid_rows,
                    grid_cols,
                    figsize=(fig_size[0] * grid_cols, fig_size[1] * grid_rows),
                )
                axs = axs.flatten()
                for ax, fig_ in zip(axs, grid_figs):
                    fig_.axes[0].get_figure().sca(fig_.axes[0])
                    for line in fig_.axes[0].get_lines():
                        xdata = line.get_xdata()
                        ydata = line.get_ydata()
                        if (
                            len(xdata) != 2
                            or len(ydata) != 2
                            or not ((xdata == [0, 1]).all() and (ydata == [0, 1]).all())
                        ):
                            ax.plot(xdata, ydata, label=line.get_label())
                    # Add dotted red diagonal line for ROC
                    if "roc" in fig_.axes[0].get_title().lower():
                        ax.plot([0, 1], [0, 1], linestyle="--")
                    ax.legend(loc="best")
                    ax.set_title(fig_.axes[0].get_title())
                    ax.set_xlabel(fig_.axes[0].get_xlabel())
                    ax.set_ylabel(fig_.axes[0].get_ylabel())
                for ax in axs[len(grid_figs) :]:
                    fig.delaxes(ax)
                plt.tight_layout()
                if save_plots:
                    filename = f"{image_prefix}_grid" if image_prefix else "grid"
                    if image_path_png:
                        os.makedirs(image_path_png, exist_ok=True)
                        plt.savefig(
                            os.path.join(image_path_png, f"{filename}.png"),
                            bbox_inches=bbox_inches,
                        )
                    if image_path_svg:
                        os.makedirs(image_path_svg, exist_ok=True)
                        plt.savefig(
                            os.path.join(image_path_svg, f"{filename}.svg"),
                            bbox_inches=bbox_inches,
                        )
                plt.show()

        if mode == "plot":
            return

    if mode == "both":
        return result


################################################################################
######################## Calculate Performance Metrics #########################
################################################################################


def calculate_metrics_for_n_var(df, n_var_list):
    """
    Calculate metrics for multiple outcomes and store the results in a DataFrame.

    This function computes a set of performance metrics for multiple binary
    classification models given the true labels and the predicted probabilities
    for each outcome. The metrics calculated include precision (positive predictive value),
    average precision, sensitivity (recall), specificity, AUC ROC, and Brier score.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the necessary columns for truth and predictions.
    n_var_list : list of int
        List of variable numbers to consider, e.g., [4, 6, 8].

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the calculated metrics for each outcome.

    Notes:
    -----
    - Precision is calculated with a threshold of 0.5 for the predicted probabilities.
    - Sensitivity is also known as recall.
    - Specificity is calculated as the recall for the negative class.
    - AUC ROC is calculated using the receiver operating characteristic curve.
    - Brier score measures the mean squared difference between predicted
      probabilities and the true binary outcomes.
    """

    outcomes = ["2_year", "5_year"]
    y_true_2_yr = df["2_year_outcome"]
    y_true_5_yr = df["5_year_outcome"]
    y_true = [y_true_2_yr, y_true_5_yr]

    preds_n_var_dict = {}
    for n_var in n_var_list:
        preds_n_var_dict[n_var] = []
        for outcome in outcomes:
            col_name = f"kfre_{n_var}var_{outcome.replace('_year', 'year')}"
            if col_name in df.columns:
                preds_n_var_dict[n_var].append(df[col_name])
            else:
                preds_n_var_dict[n_var].append(None)

    metrics_list_n_var = []

    for n_var, preds in preds_n_var_dict.items():
        for outcome, true_labels, pred_labels in zip(outcomes, y_true, preds):
            if pred_labels is not None:
                precision = precision_score(true_labels, pred_labels > 0.5)
                sensitivity = recall_score(true_labels, pred_labels > 0.5)
                specificity = recall_score(true_labels, pred_labels > 0.5, pos_label=0)
                auc_roc = roc_auc_score(true_labels, pred_labels)
                brier = brier_score_loss(true_labels, pred_labels)
                average_precision = average_precision_score(true_labels, pred_labels)

                metrics = {
                    "Precision/PPV": precision,
                    "Average Precision": average_precision,
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "AUC ROC": auc_roc,
                    "Brier Score": brier,
                    "Outcome": f"{outcome}_{n_var}_var_kfre",
                }

                metrics_list_n_var.append(metrics)

    metrics_df_n_var = pd.DataFrame(metrics_list_n_var)
    metrics_df_n_var = metrics_df_n_var.set_index("Outcome").T
    metrics_df_n_var = metrics_df_n_var.rename_axis("Metrics")

    return metrics_df_n_var
