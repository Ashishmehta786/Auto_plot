#import streamlit as st

import atexit
from tqdm import tqdm
import time
import pdfkit
from flask import Flask, render_template, request, send_file, url_for
import pandas as pd
#from flask import Flask,render_template, request
import csv
import os
#import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

#from pandas_profiling import ProfileReport
#from google.colab import files
#import os
#import glob
app = Flask(__name__, template_folder='template')

UPLOAD_FOLDER = 'csv'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
created_files = []
created_image_files = []


def delete_created_files():
  for file_path in created_files:
    os.remove(file_path)
  for file_path1 in created_image_files:
    os.remove(file_path1)


atexit.register(delete_created_files)
#will try it in future to optimize the summary
"""def handle_outliers(data, method='zscore', threshold=3):
    if method == 'zscore':
        z_scores = data.apply(lambda x: (x - x.mean()) / x.std())
        return data[(z_scores.abs() < threshold).all(axis=1)]
    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    else:
        return data

def handle_missing_values(data, method='mean'):
    if method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    elif method == 'mode':
        return data.fillna(data.mode().iloc[0])
    elif method == 'interpolation':
        return data.interpolate()
    else:
        return data here comment ends"""


@app.route('/')
def index():
  url = url_for('upload')
  file_name = request.args.get(
    'file_name', '')  # Pass the URL of the 'upload' route to the template
  return render_template("upload.html",
                         url=url,
                         file_name=file_name,
                         loading=False)


def perform_eda(data):
  # Handling Outliers and Missing Values
  # data = handle_outliers(data)
  #data = handle_missing_values(data)
  # Summary Information
  summary = {}
  data = data.dropna(axis=0)
  # Data Information
  summary['Data Information'] = data.info()

  # Summary Statistics
  summary['Summary Statistics'] = data.describe()

  # Data Correlation - Heatmap (excluding non-numeric columns)

  numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

  correlation_matrix = data[numeric_columns].corr()
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title("Correlation Matrix - Heatmap")
  plt.tight_layout()
  correlation_plot_path = "static/correlation_heatmap.svg"
  created_files.append(correlation_plot_path)
  plt.savefig(correlation_plot_path)
  plt.close()
  summary['Correlation Matrix - Heatmap'] = correlation_plot_path
  # Categorical Plots
  categorical_columns = data.select_dtypes(include='object').columns
  for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=data, palette='pastel')
    plt.title(f"{column} - Count Plot")
    plt.xticks(rotation=45)
    plt.tight_layout()
    count_plot_path = f"static/{column}_count_plot.svg"
    created_files.append(count_plot_path)
    plt.savefig(count_plot_path)
    plt.close()
    summary[f'{column} - Count Plot'] = count_plot_path

    # Count Plot Summary
    summary[f'{column} - Count Plot Summary'] = \
        f"The count plot displays the distribution of the '{column}' category. " \
        f"It shows the frequency of each category in the dataset."

  # Numerical Plots
  numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
  for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True, color='skyblue')
    plt.title(f"{column} - Histogram")
    plt.tight_layout()
    histogram_plot_path = f"static/{column}_histogram.svg"
    created_files.append(histogram_plot_path)
    plt.savefig(histogram_plot_path)
    plt.close()
    summary[f'{column} - Histogram'] = histogram_plot_path

    # Histogram Summary
    summary[f'{column} - Histogram Summary'] = \
        f"The histogram shows the distribution of '{column}' numerical data. " \
        f"It provides insights into the data's central tendency, spread, and skewness."

    # Log-Log Scaling Plot
     
    for column in numeric_columns:
      if data[column].min() > 0:
         plt.figure(figsize=(8, 6))
         plt.loglog(data[column], marker='o', linestyle='', markersize=4)
         plt.title(f"{column} - Log-Log Scaling Plot")
         plt.tight_layout()
         log_log_plot_path = f"static/{column}_log_log_scaling_plot.svg"
         created_files.append(log_log_plot_path)
         plt.savefig(log_log_plot_path)
         plt.close()
         summary[f'{column} - Log-Log Scaling Plot'] = log_log_plot_path

        # Log-Log Scaling Plot Summary
         summary[f'{column} - Log-Log Scaling Plot Summary'] = \
            f"The log-log scaling plot visualizes '{column}' numerical data on " \
            f"a logarithmic scale on both the x-axis and y-axis. It is useful for "  \
            f"visualizing relationships that might not be evident in linear plots."



    # Additional Numerical Plots
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=column, data=data, palette='pastel')
    plt.title(f"{column} - Box Plot")
    plt.tight_layout()
    box_plot_path = f"static/{column}_box_plot.svg"
    created_files.append(box_plot_path)
    plt.savefig(box_plot_path)
    plt.close()
    summary[f'{column} - Box Plot'] = box_plot_path

    # Box Plot Summary
    summary[f'{column} - Box Plot Summary'] = \
        f"The box plot displays the distribution of '{column}' numerical data through " \
        f"quartiles, showing the median, interquartile range (IQR), and potential outliers."

    plt.figure(figsize=(8, 6))
    sns.violinplot(x=column, data=data, palette='pastel')
    plt.title(f"{column} - Violin Plot")
    plt.tight_layout()
    violin_plot_path = f"static/{column}_violin_plot.svg"
    created_files.append(violin_plot_path)
    plt.savefig(violin_plot_path)
    plt.close()
    summary[f'{column} - Violin Plot'] = violin_plot_path

    # Violin Plot Summary
    summary[f'{column} - Violin Plot Summary'] = \
        f"The violin plot combines box plots and kernel density plots to show " \
        f"the distribution and density of '{column}' numerical data. It is useful " \
        f"for comparing the distribution of multiple categories."

    plt.figure(figsize=(8, 6))
    sns.kdeplot(data[column], shade=True, color='skyblue')
    plt.title(f"{column} - KDE Plot")
    plt.tight_layout()
    kde_plot_path = f"static/{column}_kde_plot.svg"
    created_files.append(kde_plot_path)
    plt.savefig(kde_plot_path)
    plt.close()
    summary[f'{column} - KDE Plot'] = kde_plot_path

    # KDE Plot Summary
    summary[f'{column} - KDE Plot Summary'] = \
        f"The KDE plot shows the probability density function of '{column}' numerical data. " \
        f"It provides a smooth estimate of the underlying data distribution."

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=data.index, y=data[column])
    plt.title(f"{column} - Line Plot")
    plt.tight_layout()
    line_plot_path = f"static/{column}_line_plot.svg"
    plt.savefig(line_plot_path)
    plt.close()
    summary[f'{column} - Line Plot'] = line_plot_path

    # Line Plot Summary
    summary[f'{column} - Line Plot Summary'] = \
        f"The line plot displays the trend or relationship between the index and '{column}' " \
        f"numerical data. It is useful for identifying patterns and trends in time series data."
    created_files.append(line_plot_path)

    # Additional Categorical Plots

    for column in categorical_columns:

      plt.figure(figsize=(8, 6))
      sns.barplot(x=data[column].value_counts().index,
                  y=data[column].value_counts(),
                  palette='pastel')
      plt.title(f"{column} - Bar Plot")
      plt.xticks(rotation=45)
      plt.tight_layout()
      bar_plot_path = f"static/{column}_bar_plot.svg"
      plt.savefig(bar_plot_path)
      plt.close()
      summary[f'{column} - Bar Plot'] = bar_plot_path

      # Bar Plot Summary
      summary[f'{column} - Bar Plot Summary'] = \
          f"The bar plot displays the distribution of '{column}' categorical data using bars. " \
          f"It is useful for comparing the frequency or count of categories across different groups."
  summary_dataframes = {}
  summary_dataframes['Summary Statistics'] = data.describe()

  # Convert dataframes to HTML tables
  summary_tables = {}
  for table_name, df in summary_dataframes.items():
    summary_tables[table_name] = df.to_html(
      classes='table table-striped table-bordered')
    created_image_files.append(bar_plot_path)

  print(summary, summary_tables)

  return summary, summary_tables


@app.route('/upload', methods=['POST', 'GET'])
def upload():
  if 'file' not in request.files:
    return 'No file found'

  file = request.files['file']

  if file.filename == '':
    return 'No selected file'

  if file:
    loading = True
    # Save the file to the uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load the data from the uploaded CSV file
    data = pd.read_csv(file_path)
    summary, summary_tables = perform_eda(data)
    loading = False
    return render_template("eda_summary.html",
                           summary=summary,
                           summary_tables=summary_tables,
                           loading=False,
                           file_name=file.filename)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=81)
