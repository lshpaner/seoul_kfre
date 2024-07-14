import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import random  # for generating random numbers and performing random operations
import os
import sys
import warnings

################################################################################
############################# Path Directories #################################


def ensure_directory(path):
    """Ensure that the directory exists. If not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory exists: {path}")


################################################################################
######################## Generate Random Patient IDs ###########################
################################################################################


def add_ids(
    df,
    column_name="Patient_ID",
    seed=None,
):
    """
    Add a column of unique, 9-digit IDs to the dataframe.

    This function sets a random seed and then generates a 9-digit ID for
    each row in the dataframe. The new IDs are added as a new column with
    the specified column name, which is placed as the first column in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add IDs to.
        column_name (str): The name of the new column for the IDs.
        seed (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
        pd.DataFrame: The updated dataframe with the new ID column.
    """
    random.seed(seed)

    # Generate a list of unique IDs
    ids = ["".join(random.choices("0123456789", k=9)) for _ in range(len(df))]

    # Create a new column in df for these IDs
    df[column_name] = ids

    # Make the new ID column the first column and set it to index
    df = df.set_index(column_name)

    return df


################################################################################


def strip_trailing_period(
    df,
    column_name,
):
    """
    Strip the trailing period from floats in a specified column of a DataFrame, if present.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the column to be processed.

    column_name : str
        The name of the column containing floats with potential trailing periods.

    Returns:
    --------
    pd.DataFrame
        The updated DataFrame with the trailing periods removed from the specified column.
    """

    def fix_value(value):
        value_str = str(value)
        if value_str.endswith("."):
            value_str = value_str.rstrip(".")
        return float(value_str)

    df[column_name] = df[column_name].apply(fix_value)

    return df


################################################################################
########################### Standardized Dates #################################
################################################################################


# Function to parse and standardize date strings based on the new rule
def parse_date_with_rule(date_str):
    """
    Parse and standardize date strings based on the provided rule.

    This function takes a date string and standardizes it to the ISO 8601 format
    (YYYY-MM-DD). It assumes dates are provided in either day/month/year or
    month/day/year format. The function first checks if the first part of the
    date string (day or month) is greater than 12, which unambiguously indicates
    a day/month/year format. If the first part is 12 or less, the function
    attempts to parse the date as month/day/year, falling back to day/month/year
    if the former raises a ValueError due to an impossible date (e.g., month
    being greater than 12).

    Parameters:
        date_str (str): A date string to be standardized.

    Returns:
        str: A standardized date string in the format YYYY-MM-DD.

    Raises:
        ValueError: If date_str is in an unrecognized format or if the function
        cannot parse the date.
    """
    parts = date_str.split("/")
    # If the first part is greater than 12, it can only be a day, thus d/m/Y
    if int(parts[0]) > 12:
        return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
    # Otherwise, try both formats where ambiguity exists
    else:
        try:
            return datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")
        except ValueError:
            return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")


################################################################################
############################ Data Types Report #################################
################################################################################


def data_types(df):
    """
    This function provides a data types report on every column in the dataframe,
    showing column names, column data types, number of nulls, and percentage
    of nulls, respectively.
    Inputs:
        df: dataframe to run the datatypes report on
    Outputs:
        dat_type: report saved out to a dataframe showing column name,
                  data type, count of null values in the dataframe, and
                  percentage of null values in the dataframe
    """
    # Features' Data Types and Their Respective Null Counts
    dat_type = df.dtypes

    # create a new dataframe to inspect data types
    dat_type = pd.DataFrame(dat_type)

    # sum the number of nulls per column in df
    dat_type["Null_Values"] = df.isnull().sum()

    # reset index w/ inplace = True for more efficient memory usage
    dat_type.reset_index(inplace=True)

    # percentage of null values is produced and cast to new variable
    dat_type["perc_null"] = round(dat_type["Null_Values"] / len(df) * 100, 0)

    # columns are renamed for a cleaner appearance
    dat_type = dat_type.rename(
        columns={
            0: "Data Type",
            "index": "Column/Variable",
            "Null_Values": "# of Nulls",
            "perc_null": "Percent Null",
        }
    )

    return dat_type


################################################################################
########################### Summarize All Combinations #########################
################################################################################


def summarize_all_combinations(
    df,
    variables,
    data_path,
    data_name,
    min_length=2,
):
    """
    Generates summary tables for all possible combinations of the specified
    variables in the dataframe and saves them to an Excel file.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - variables (list): List of unique variables to generate combinations.
    - data_path (str): Path where the output Excel file will be saved.
    - data_name (str): Name of the output Excel file.
    - min_length (int): Minimum length of combinations to generate. Defaults to 2.

    Returns:
    - summary_tables (dict): Dictionary of summary tables.
    - all_combinations (list): List of all generated combinations.
    """
    summary_tables = {}
    grand_total = len(df)
    all_combinations = []

    # Generate all possible combinations of the unique variables
    for i in range(min_length, len(variables) + 1):
        for combination in combinations(variables, i):
            all_combinations.append(combination)
            for col in combination:
                df[col] = df[col].astype(str)

            # Count
            count_df = df.groupby(list(combination)).size().reset_index(name="Count")
            # Proportion as percentage of grand total
            count_df["Proportion"] = (count_df["Count"] / grand_total * 100).fillna(0)

            summary_tables[tuple(combination)] = count_df

    # Save each dataframe as a separate sheet in an Excel file
    file_path = f"{data_path}/{data_name}"
    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        for combination, table in summary_tables.items():
            sheet_name = "_".join(combination)[
                :31
            ]  # Excel sheet names must be <= 31 characters
            table.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Data saved to {file_path}")

    return summary_tables, all_combinations


################################################################################
################################ Cross-Tab Plot ################################
################################################################################


def crosstab_plot(
    df,
    outcome,
    sub1,
    sub2,
    x,
    y,
    list_name,
    col1,
    bbox_to_anchor,
    w_pad,
    h_pad,
    item1=None,
    item2=None,
    label1=None,
    label2=None,
    crosstab_option=True,
    image_path_png=None,
    image_path_svg=None,
    image_filename=None,
    tight_layout=True,
    bbox_inches=None,
):
    """
    Generates a series of crosstab plots to visualize the relationship between
    an outcome variable and several categorical variables within a dataset. Each
    subplot represents the distribution of outcomes for a specific categorical
    variable, allowing for comparisons across categories.

    The subplot grid, plot size, legend placement, and subplot padding are
    customizable. The function can create standard or normalized crosstab plots
    based on the 'crosstab_option' flag.

    Parameters:
    - df: The DataFrame to pass in.
    - sub1, sub2 (int): The number of rows and columns in the subplot grid.
    - x, y (int): Width & height of ea. subplot, affecting the overall fig. size.
    - list_name (list[str]): A list of strings rep. the column names to be plotted.
    - label1, label2 (str): Labels for the x-axis categories, corresponding to
                            the unique values in the 'outcome' variable of the
                            dataframe.
    - col1 (str): The column name in the dataframe for which custom legend
                  labels are desired.
    - item1, item2 (str): Custom legend labels for the plot corresponding to 'col1'.
    - bbox_to_anchor (tuple): A tuple of (x, y) coordinates to anchor the legend to a
                              specific point within the axes.
    - w_pad, h_pad (float): The amount of width and height padding (space)
                            between subplots.
    - crosstab_option (bool, optional): If True, generates standard crosstab
                                        plots. If False, generates normalized
                                        crosstab plots, which are useful for
                                        comparing distributions across groups
                                        with different sizes.
    - image_path_png (str): Path to save PNG files.
    - image_path_svg (str): Path to save SVG files.
    - image_filename (str): Base filename for the output image.
    - bbox_inches (str): specify tightness of bbox_inches for visibility.

    The function creates a figure with the specified number of subplots laid out
    in a grid, plots the crosstabulation data as bar plots within each subplot,
    and then adjusts the legend and labels accordingly. It uses a tight layout
    with specified padding to ensure that subplots are neatly arranged without
    overlapping elements.
    """

    fig, axes = plt.subplots(sub1, sub2, figsize=(x, y))
    for item, ax in zip(list_name, axes.flatten()):
        if crosstab_option:
            # Set a fixed number of ticks for raw data
            ax.set_ylabel("Frequency"),
            crosstab_data = pd.crosstab(df[outcome], df[item])
            crosstab_data.plot(
                kind="bar",
                stacked=True,
                rot=0,
                ax=ax,
                color=["#00BFC4", "#F8766D"],
            )

        else:
            # Set a fixed number of ticks for percentage data
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: "{:.2f}".format(y))
            )
            ax.set_ylabel("Percentage"),
            # Computing normalized crosstabulation
            crosstab_data = pd.crosstab(
                df[outcome],
                df[item],
                normalize="index",
            )
            crosstab_data.plot(
                kind="bar",
                stacked=True,
                rot=0,
                ax=ax,
                color=["#00BFC4", "#F8766D"],
            )

        new_labels = [label1, label2]
        ax.set_xticklabels(new_labels)
        # new_legend = ["Not Obese", "Obese"]
        # ax.legend(new_legend)
        ax.set_title(f"{outcome} vs. {item}")
        ax.set_xlabel("Outcome")
        # Dynamically setting legend labels
        # Check if the current column is 'Sex' for custom legend labels
        if item == col1:
            legend_labels = [item1, item2]
        else:
            # Dynamically setting legend labels for other columns
            legend_labels = ["NOT {}".format(item), "{}".format(item)]

        # Updating legend with custom labels
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,
        )

    if tight_layout:
        plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # Save files if paths are provided
    if image_path_png and image_filename:
        plt.savefig(
            os.path.join(image_path_png, f"{image_filename}.png"),
            bbox_inches=bbox_inches,
        )
    if image_path_svg and image_filename:
        plt.savefig(
            os.path.join(image_path_svg, f"{image_filename}.svg"),
            bbox_inches=bbox_inches,
        )

    plt.show()


################################################################################
############################# Box Plots Assortment #############################
################################################################################


def create_metrics_boxplots(
    df_eda,
    metrics_list,
    metrics_boxplot_comp,
    n_rows,
    n_cols,
    image_path_png,
    image_path_svg,
    save_individual=True,
    save_grid=True,
    save_both=False,
):
    """
    Create and save individual boxplots, an entire grid of boxplots, or both for
    given metrics and comparisons.

    Parameters:
    - df_eda: DataFrame containing the data.
    - metrics_list: List of metric names (columns in df_eda) to plot.
    - metrics_boxplot_comp: List of comparison categories (columns in df_eda).
    - n_rows: Number of rows in the subplot grid.
    - n_cols: Number of columns in the subplot grid.
    - image_path_png: Directory path to save .png images.
    - image_path_svg: Directory path to save .svg images.
    - save_individual: Boolean, True if saving each subplot as an individual file.
    - save_grid: Boolean, True if saving the entire grid as one image.
    - save_both: Boolean, True if saving both individual and grid images.
    """
    # Ensure the directories exist
    os.makedirs(image_path_png, exist_ok=True)
    os.makedirs(image_path_svg, exist_ok=True)

    if save_both:
        save_individual = True
        save_grid = True

    # Save individual plots if required
    if save_individual:
        for met_comp in metrics_boxplot_comp:
            for met_list in metrics_list:
                plt.figure(figsize=(6, 4))  # Adjust the size as needed
                sns.boxplot(x=df_eda[met_comp], y=df_eda[met_list])
                plt.title(f"Distribution of {met_list} by {met_comp}")
                plt.xlabel(met_comp)
                plt.ylabel(met_list)
                safe_met_list = (
                    met_list.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_per_")
                )
                filename_png = f"{safe_met_list}_by_{met_comp}.png"
                filename_svg = f"{safe_met_list}_by_{met_comp}.svg"
                plt.savefig(
                    os.path.join(image_path_png, filename_png), bbox_inches="tight"
                )
                plt.savefig(
                    os.path.join(image_path_svg, filename_svg), bbox_inches="tight"
                )
                plt.close()

    # Save the entire grid if required
    if save_grid:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            if i < len(metrics_list) * len(metrics_boxplot_comp):
                met_comp = metrics_boxplot_comp[i // len(metrics_list)]
                met_list = metrics_list[i % len(metrics_list)]
                sns.boxplot(x=df_eda[met_comp], y=df_eda[met_list], ax=ax)
                ax.set_title(f"Distribution of {met_list} by {met_comp}")
                ax.set_xlabel(met_comp)
                ax.set_ylabel(met_list)
            else:
                ax.set_visible(False)

        plt.tight_layout()
        fig.savefig(
            os.path.join(image_path_png, "all_boxplot_comparisons.png"),
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(image_path_svg, "all_boxplot_comparisons.svg"),
            bbox_inches="tight",
        )
        plt.show()  # show the plot(s)
        plt.close(fig)


################################################################################
############################# Stacked Bar Plot #################################
################################################################################


def stacked_crosstab_plot(
    x,
    y,
    p,
    df,
    col,
    func_col,
    legend_labels_list,
    title,
    file_prefix,  # New parameter for filename prefix
    kind="bar",
    width=0.9,
    rot=0,
    ascending=True,
    string=None,
    custom_order=None,
    image_path_png=None,
    image_path_svg=None,
    img_string=None,
    save_formats=None,
    color=None,
    output="both",  # New parameter to specify output type
    return_dict=False,  # New parameter to control whether to return the dictionary
):
    """
    Generates pairs of stacked bar plots for specified columns against
    ground truth columns, with the first plot showing absolute distributions
    and the second plot showing normalized distributions. Offers customization
    options for plot titles, colors, and more. Also stores and displays crosstabs.

    Parameters:
    - x (int): The width of the figure.
    - y (int): The height of the figure.
    - p (int): The padding between the subplots.
    - df (DataFrame): The pandas DataFrame containing the data.
    - col (str): The name of the column in the DataFrame to be analyzed.
    - func_col (list): List of ground truth columns to be analyzed.
    - legend_labels_list (list): List of legend labels for each ground truth
      column.
    - title (list): List of titles for the plots.
    - file_prefix (str): Prefix for the filename.
    - kind (str, optional): The kind of plot to generate (e.g., 'bar', 'barh').
      Defaults to 'bar'.
    - width (float, optional): The width of the bars in the bar plot. Defaults
      to 0.9.
    - rot (int, optional): The rotation angle of the x-axis labels. Defaults
      to 0.
    - ascending (bool, optional): Determines the sorting order of the DataFrame
      based on the 'col'. Defaults to True.
    - string (str, optional): Descriptive string to include in the title of the
      plots.
    - custom_order (list, optional): Specifies a custom order for the categories
      in the 'col'. If provided, the DataFrame is sorted according to this order.
    - image_path_png (str, optional): Directory path where generated PNG plot
      images will be saved.
    - image_path_svg (str, optional): Directory path where generated SVG plot
      images will be saved.
    - img_string (str, optional): Filename for the saved plot images.
    - save_formats (list, optional): List of file formats to save the plot
      images in.
    - color (list, optional): List of colors to use for the plots. If not
      provided, a default color scheme is used.
    - output (str, optional): Specify the output type: "plots_only",
      "crosstabs_only", or "both". Defaults to "both".
    - return_dict (bool, optional): Specify whether to return the crosstabs
      dictionary. Defaults to False.

    Returns:
    - crosstabs_dict (dict): Dictionary of crosstabs DataFrames if return_dict
      is True.
    - None: If return_dict is False.
    """

    # Initialize the dictionary to store crosstabs
    crosstabs_dict = {}

    # Default color settings
    if color is None:
        color = ["#00BFC4", "#F8766D"]  # Default colors

    # Loop through each condition and create the plots
    for truth, legend, tit in zip(func_col, legend_labels_list, title):
        func_col_filename_png = os.path.join(
            image_path_png, f"{file_prefix}_{truth}.png"
        )
        func_col_filename_svg = os.path.join(
            image_path_svg, f"{file_prefix}_{truth}.svg"
        )

        image_path = {"png": func_col_filename_png, "svg": func_col_filename_svg}

        # Setting custom order if provided
        if custom_order:
            df[col] = pd.Categorical(df[col], categories=custom_order, ordered=True)
            df.sort_values(by=col, inplace=True)

        # Crosstabulation of column of interest and ground truth
        crosstabdest = pd.crosstab(df[col], df[truth])

        # Normalized crosstabulation
        crosstabdestnorm = crosstabdest.div(crosstabdest.sum(1), axis=0)

        # Create the full crosstab with percentages
        crosstab_full = pd.crosstab(
            df[col], df[truth], margins=True, margins_name="Total"
        )
        if len(legend) == 2:
            crosstab_full = crosstab_full.rename(columns={0: legend[0], 1: legend[1]})
            crosstab_full[legend[1] + "_%"] = (
                round(crosstab_full[legend[1]] / crosstab_full["Total"], 2) * 100
            )
            crosstab_full[legend[0] + "_%"] = (
                round(crosstab_full[legend[0]] / crosstab_full["Total"], 2) * 100
            )
            crosstab_full["Total_%"] = (
                crosstab_full[legend[1] + "_%"] + crosstab_full[legend[0] + "_%"]
            )

        # Store the crosstab in the dictionary
        crosstabs_dict[tit] = crosstab_full

        if output in ["plots_only", "both"]:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(x, y))
            fig.tight_layout(w_pad=5, pad=p, h_pad=5)

            # Title construction logic
            title1 = f"Prevalence of {tit} by {col.replace('_', ' ').title()}"
            title2 = (
                f"Prevalence of {tit} by {col.replace('_', ' ').title()} (Normalized)"
            )

            xlabel1 = xlabel2 = f"{col.replace('_', ' ').title()}"
            ylabel1 = "Count"
            ylabel2 = "Density"

            # Plotting the first stacked bar graph
            crosstabdest.plot(
                kind=kind,
                stacked=True,
                title=title1,
                ax=axes[0],
                color=color,
                width=width,
                rot=rot,
                fontsize=12,
            )
            axes[0].set_title(title1, fontsize=12)
            axes[0].set_xlabel(xlabel1, fontsize=12)
            axes[0].set_ylabel(ylabel1, fontsize=12)
            axes[0].legend(legend, fontsize=12)

            # Plotting the second, normalized stacked bar graph
            crosstabdestnorm.plot(
                kind=kind,
                stacked=True,
                title=title2,
                ylabel="Density",
                ax=axes[1],
                color=color,
                width=width,
                rot=rot,
                fontsize=12,
            )
            axes[1].set_title(label=title2, fontsize=12)
            axes[1].set_xlabel(xlabel2, fontsize=12)
            axes[1].set_ylabel(ylabel2, fontsize=12)
            axes[1].legend(legend, fontsize=12)

            fig.align_ylabels()

            if img_string and save_formats and isinstance(image_path, dict):
                for save_format in save_formats:
                    if save_format in image_path:
                        full_path = image_path[save_format]
                        plt.savefig(full_path, bbox_inches="tight")

            plt.show()

    # Display the crosstabs if output includes crosstabs
    if output in ["crosstabs_only", "both"]:
        for col, crosstab in crosstabs_dict.items():
            print(f"Crosstab for {col}:")
            display(crosstab)

    # Return the crosstabs_dict only if return_dict is True
    if return_dict:
        return crosstabs_dict


################################################################################
############################ KDE Distribution Plots ############################
################################################################################


def kde_distributions(
    df,
    dist_list,
    x,
    y,
    kde=True,
    n_rows=1,
    n_cols=1,
    w_pad=1.0,
    h_pad=1.0,
    text_wrap=50,
    image_path_png=None,
    image_path_svg=None,
    image_filename=None,
    bbox_inches=None,
    vars_of_interest=None,  # List of variables of interest
    single_var_image_path_png=None,
    single_var_image_path_svg=None,
    single_var_image_filename=None,
    y_axis="count",  # Parameter to control y-axis ('count' or 'density')
    plot_type="both",  # Parameter to control the plot type ('hist', 'kde', or 'both')
):
    """
    Generate KDE or histogram distribution plots for specified columns in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.

    dist_list : list of str
        List of column names for which to generate distribution plots.

    x : int or float
        Width of the overall figure.

    y : int or float
        Height of the overall figure.

    kde : bool, optional (default=True)
        Whether to include KDE plots on the histograms.

    n_rows : int, optional (default=1)
        Number of rows in the subplot grid.

    n_cols : int, optional (default=1)
        Number of columns in the subplot grid.

    w_pad : float, optional (default=1.0)
        Width padding between subplots.

    h_pad : float, optional (default=1.0)
        Height padding between subplots.

    text_wrap : int, optional (default=50)
        Maximum width of the title text before wrapping.

    image_path_png : str, optional
        Directory path to save the PNG image of the overall distribution plots.

    image_path_svg : str, optional
        Directory path to save the SVG image of the overall distribution plots.

    image_filename : str, optional
        Filename to use when saving the overall distribution plots.

    bbox_inches : str, optional
        Bounding box to use when saving the figure. For example, 'tight'.

    vars_of_interest : list of str, optional
        List of column names for which to generate separate distribution plots.

    single_var_image_path_png : str, optional
        Directory path to save the PNG images of the separate distribution plots.

    single_var_image_path_svg : str, optional
        Directory path to save the SVG images of the separate distribution plots.

    single_var_image_filename : str, optional
        Filename to use when saving the separate distribution plots.
        The variable name will be appended to this filename.

    y_axis : str, optional (default='count')
        The type of y-axis to display ('count' or 'density').

    plot_type : str, optional (default='both')
        The type of plot to generate ('hist', 'kde', or 'both').

    Returns:
    --------
    None
    """

    if not dist_list:
        print("Error: No distribution list provided.")
        return

    y_axis = y_axis.lower()
    if y_axis not in ["count", "density"]:
        raise ValueError('y_axis can either be "count" or "density"')

    plot_type = plot_type.lower()
    if plot_type not in ["hist", "kde", "both"]:
        raise ValueError('plot_type can either be "hist", "kde", or "both"')

    # Calculate the number of plots
    num_plots = len(dist_list)
    total_slots = n_rows * n_cols

    # Create subplots grid
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(x, y))

    # Flatten the axes array to simplify iteration
    axes = axes.flatten()

    # Iterate over the provided column list and corresponding axes
    for ax, col in zip(axes[:num_plots], dist_list):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Wrap the title if it's too long
            title = f"Distribution of {col}"

            if plot_type == "hist" or plot_type == "both":
                sns.histplot(
                    df[col],
                    kde=False if plot_type == "hist" else kde,
                    ax=ax,
                    stat="density" if y_axis == "density" else "count",
                )
            if plot_type == "kde":
                sns.kdeplot(df[col], ax=ax, fill=True)
            elif plot_type == "both":
                sns.kdeplot(df[col], ax=ax)

            ax.set_ylabel("Density" if y_axis == "density" else "Count")
            ax.set_title("\n".join(textwrap.wrap(title, width=text_wrap)))

    # Hide any remaining axes
    for ax in axes[num_plots:]:
        ax.axis("off")

    # Adjust layout with specified padding
    plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # Save files if paths are provided
    if image_path_png and image_filename:
        plt.savefig(
            os.path.join(image_path_png, f"{image_filename}.png"),
            bbox_inches=bbox_inches,
        )
    if image_path_svg and image_filename:
        plt.savefig(
            os.path.join(image_path_svg, f"{image_filename}.svg"),
            bbox_inches=bbox_inches,
        )
    plt.show()

    # Generate separate plots for each variable of interest if provided
    if vars_of_interest:
        for var in vars_of_interest:
            fig, ax = plt.subplots(figsize=(x, y))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                title = f"Distribution of {var}"

                if plot_type == "hist" or plot_type == "both":
                    sns.histplot(
                        df[var],
                        kde=False if plot_type == "hist" else kde,
                        ax=ax,
                        stat="density" if y_axis == "density" else "count",
                    )
                if plot_type == "kde":
                    sns.kdeplot(df[var], ax=ax, fill=True)
                elif plot_type == "both":
                    sns.kdeplot(df[var], ax=ax)

                ax.set_ylabel("Density" if y_axis == "density" else "Count")
                ax.set_title("\n".join(textwrap.wrap(title, width=text_wrap)))

            plt.tight_layout()

            # Save files for the variable of interest if paths are provided
            if single_var_image_path_png and single_var_image_filename:
                plt.savefig(
                    os.path.join(
                        single_var_image_path_png,
                        f"{single_var_image_filename}_{var}.png",
                    ),
                    bbox_inches=bbox_inches,
                )
            if single_var_image_path_svg and single_var_image_filename:
                plt.savefig(
                    os.path.join(
                        single_var_image_path_svg,
                        f"{single_var_image_filename}_{var}.svg",
                    ),
                    bbox_inches=bbox_inches,
                )
            plt.close(
                fig
            )  # Close the figure after saving to avoid displaying it multiple times
