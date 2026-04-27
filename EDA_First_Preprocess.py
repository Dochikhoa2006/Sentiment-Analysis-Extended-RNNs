
from pyspark.sql import SparkSession
from pyspark.sql import functions 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import squarify as sq
import joblib
import re

spark = SparkSession.builder.master ("local[*]").appName ("ParquetProcessing").getOrCreate ()
dataset = spark.read.parquet ('/Users/chikhoado/Desktop/PROJECTS/Sentiment Analyzer/reviews.parquet')

def DataFrame ():

    print ("\n-------------- Dataset Structure --------------\n")
    dataset.printSchema ()

    rows = dataset.count ()
    columns = len (dataset.columns)
    print (f'+ Number of Reviews Sent: {rows}\n+ Number of Features: {columns}')

    print ("\n-------------- How First-Five Datapoint Look Like --------------\n")
    dataset.show (5)

    print ("\n-------------- Check Missing Value(s) --------------\n")
    dataset.describe ().show ()

    print ("\n-------------- Check Unique Value(s) --------------\n")
    dataset.select ('review').distinct ().show ()
    dataset.select ('star').distinct ().show ()
    dataset.select ('date').distinct ().show ()
    dataset.select ('package_name').distinct ().show ()

def ratings_distribution ():

    ratings = dataset.select ('star').toPandas ()
    ratings = ratings['star'].values 
    
    plt.figure (figsize = (12, 6))
    sns.histplot (ratings, bins = 15, kde = True)
    plt.gca ().text(0.05, 0.9, f'Mean: 4.33\nStandard Deviation: 1.38', transform = plt.gca ().transAxes, bbox = dict (boxstyle = 'round'))
    plt.title ("Distribution of Scorings")
    plt.xlabel ("Ratings (1-5)")
    plt.ylabel ("Number of Ratings")
    plt.grid (True)

    plt.savefig ('Scorings_Distribution.png')
    plt.show ()

def timestamps_density ():
    
    start_date_amazon_appstore = '2015-05-21 22:58:40'
    upto_date_amazon_appstore = '2024-05-01 18:17:28'

    temp1 = dataset.filter (functions.col ('date').between (start_date_amazon_appstore, upto_date_amazon_appstore)) 
    temp2 = temp1.groupBy (functions.window ('date', '180 days'))
    temp3 = temp2.agg (functions.count ('*').alias ('count'))
    temp4 = temp3.orderBy ('window.start')
    temp5 = temp4.select (functions.col ('window.start').cast ('string'), 'count')
    final_date_filter = temp5.collect ()

    every_6_months = []
    density_every_6_months = []

    for row in final_date_filter:

        _date_, total_nums_point = row[0], row[1]
        _date_ = re.split (r' ', _date_)
        every_6_months.append (_date_[0])
        density_every_6_months.append (total_nums_point)

    every_6_months[0] = '2015-05-21'
    every_6_months[-1] = '2024-05-01'

    plt.figure (figsize = (15, 6))
    plt.bar (every_6_months, density_every_6_months, color = 'skyblue')
    plt.title ('Scorings Density Every 6 Months')
    plt.ylabel ('Number of Ratings')
    plt.xlabel ('Date')
    plt.grid (True)

    plt.gcf ().autofmt_xdate ()
    plt.tight_layout ()
    plt.savefig ('Scorings_Density_Every_6_Months.png')
    plt.show ()

def application_density ():

    temp1 = dataset.groupBy ('package_name')
    temp2 = temp1.count ().orderBy (functions.col ('count').desc ())
    application_ratings_nums = temp2.collect ()

    application_name = []
    nums_ratings = []
    sum = 0

    for index, row in enumerate (application_ratings_nums):
        if index < 50:
            application_name.append (f'Nums score = {str (row[1])}')
            nums_ratings.append (row[1])
        else:
            sum += row[1]
    sum = int (sum / (len (application_ratings_nums) - 50))
    application_name.append (f'Average\nOther\nApps\n={str(sum)}')
    nums_ratings.append (sum)

    plt.figure (figsize = (15, 8))
    sq.plot (sizes = nums_ratings, label = application_name)
    plt.title ('Top 50 Density of Applications Scored')
    plt.axis ('off')

    plt.tight_layout ()
    plt.savefig ('Density_Applications_Scored.png')
    plt.show ()

def missing_values ():

    global dataset
    dataset = dataset.dropna (subset = 'review')

def drop_irrelevant_column_and_reduce_output ():

    global dataset
    dataset = dataset.select (['review', 'star']).toPandas ()
    dataset['star'] = dataset['star'] - 1


DataFrame ()
ratings_distribution ()
timestamps_density ()
application_density ()
missing_values ()
drop_irrelevant_column_and_reduce_output ()

joblib.dump (dataset, 'setup_dataset.pkl')






