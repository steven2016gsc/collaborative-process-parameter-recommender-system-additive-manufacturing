import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import pyspark
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.utils.notebook_utils import is_jupyter
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.utils.notebook_utils import store_metadata
from pyspark.sql.functions import round
from tqdm import trange



print(f"System version: {sys.version}")
print("Spark version: {}".format(pyspark.__version__))

spark = start_or_get_spark("ALS PySpark", memory="16g")
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")

COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
# Use the directory where this script is located
PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
DATA_DIR = os.path.join(PATH, "data")
RESULTS_DIR = os.path.join(PATH, "results")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)


schema = StructType(
    (
        StructField(COL_USER, IntegerType()),
        StructField(COL_ITEM, IntegerType()),
        StructField(COL_RATING, FloatType()),
    )
)

header = {
    "userCol": COL_USER,
    "itemCol": COL_ITEM,
    "ratingCol": COL_RATING,
}


als = ALS(
    rank=3,
    maxIter=15,
    implicitPrefs=False,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=True,
    seed=42,
    **header
)
regrets_separate = [0]

# MC - non-collab
als_sub = ALS(
    rank=3,
    maxIter=15,
    implicitPrefs=False,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=True,
    seed=42,
    **header
)

data_np = np.genfromtxt(os.path.join(DATA_DIR,'nU_exp.csv'), delimiter=',')


row_size, col_size = data_np.shape[0], data_np.shape[1]
data_train = pd.DataFrame(columns=[1,2,3])

train_row_idx = 0
test_row_idx = 0
data_test = pd.DataFrame(columns=[1,2,3])
small_matrix_rows = 5 # how many rows in the small matrix

missing_N = int(col_size * 0.55) # missing rate is 55%
np.random.seed(128)
for row in trange(row_size, desc='Train/test Episodes'):
    data_train_solo = pd.DataFrame(columns=[1,2,3])
    data_test_solo = pd.DataFrame(columns=[1,2,3])
    train_solo_ind = 0
    test_solo_ind = 0
    full_col_set = [*range(col_size)]
    
    cand_set=[]
    rot_set_1 = [5,12,19,26,33]
    rot_set_2 = [13,20,27,34,6]
    if row==0 or row==5:
        cand_set = [0, 8, 16, 24, 32]
    elif row==1 or row==6:
        cand_set = [1, 9, 17, 25, 28]
    elif row==2 or row==7:
        cand_set = [2, 10, 18, 21, 29]
    elif row==3 or row==8:
        cand_set = [3, 11, 14, 22, 30]
    elif row==4 or row==9:
        cand_set = [4, 7, 15, 23, 31]
    diag_set = [i*(col_size//small_matrix_rows)+i for i in range(5)] # all the same col
    
    diag_set.append(5) # all the same col
    diag_set.append(13) # all the same col
    set_to_be_missed = list(set(full_col_set) - set(diag_set))
    
    test_id = np.random.choice(set_to_be_missed,missing_N,replace=False).tolist()
    for col in range(col_size):
        if col not in test_id:
            data_train.loc[train_row_idx, 1] = row
            data_train.loc[train_row_idx, 2] = col
            data_train.loc[train_row_idx, 3] = data_np[row,col]
            data_train_solo.loc[train_solo_ind, 1] = col//small_matrix_rows
            data_train_solo.loc[train_solo_ind, 2] = col%small_matrix_rows
            data_train_solo.loc[train_solo_ind, 3] = data_np[row,col]
            train_row_idx += 1
            train_solo_ind += 1
        else:
            data_test.loc[test_row_idx, 1] = row
            data_test.loc[test_row_idx, 2] = col
            data_test.loc[test_row_idx, 3] = data_np[row,col]
            data_test_solo.loc[test_solo_ind, 1] = col//small_matrix_rows
            data_test_solo.loc[test_solo_ind, 2] = col%small_matrix_rows
            data_test_solo.loc[test_solo_ind, 3] = data_np[row,col]
            test_row_idx += 1
            test_solo_ind += 1
    data_train_solo.to_csv(PATH+'new_rating_data_train_y_'+str(row+1)+'.csv',index=False,header=False)
    data_test_solo.to_csv(PATH+'new_rating_data_test_y_'+str(row+1)+'.csv',index=False,header=False)
         

data_train.to_csv(PATH+'new_rating_data_train_y.csv',index=False,header=False)
data_test.to_csv(PATH+'new_rating_data_test_y.csv',index=False,header=False)
train = spark.read.csv(path=PATH+'new_rating_data_train_y.csv',schema=schema)
test = spark.read.csv(path=PATH+'new_rating_data_test_y.csv',schema=schema)


cumu_regrets_largest_colab = np.zeros(shape=(row_size, missing_N))
cumu_regrets_largest_solo = np.zeros(shape=(row_size, missing_N))

first_find_minimal_tab = missing_N*np.ones(shape=(10,4))
import random

#batch setting
for iter_i in trange(missing_N, desc='Global Train/test Episodes'):
    # global train and test
    model = als.fit(train)
    prediction = model.transform(test)
        
    
    # transform predictions to different printers
    # SQL based filtering
    user_factors = model.userFactors
    item_factors = model.itemFactors
    user_factors_np = np.array(user_factors.select("features").rdd.map(lambda row: row[0]).collect())
    item_factors_np = np.array(item_factors.select("features").rdd.map(lambda row: row[0]).collect())
    id_index = np.array(user_factors.select("id").rdd.map(lambda row: row[0]).collect())
    id_index_item = np.array(item_factors.select("id").rdd.map(lambda row: row[0]).collect())
    
    matrix_X = np.zeros(shape=user_factors_np.shape)
    for j in range(matrix_X.shape[0]):
        matrix_X[int(id_index[j]),:]=user_factors_np[j,:]
    matrix_Y = np.zeros(shape=item_factors_np.shape)
    for j in range(matrix_Y.shape[0]):
        matrix_Y[int(id_index_item[j]),:]=item_factors_np[j,:]
    
    test_matrix = np.array(test.select(["UserId","MovieId","Rating"]).rdd.map(lambda x:(int(x[0]),int(x[1]),x[2])).collect())
    test_dict = {}
    for test_i in range(test_matrix.shape[0]):
        printer_i, condition_i, rating_i = test_matrix[test_i,:]
        if printer_i not in test_dict.keys():
            test_dict[int(printer_i)] = [(int(condition_i),rating_i)]
        else:
            test_dict[int(printer_i)].append((int(condition_i),rating_i))
    
    
    
    candidates = []
    
    for printer_i in range(row_size):
        nutility = []
        for test_printer_i,_ in test_dict[printer_i]:
            nutility.append(np.abs(matrix_X[printer_i] @ matrix_Y[test_printer_i])) # largest prediction
        user,movie,rating=printer_i,test_dict[printer_i][np.argmin(nutility)][0],test_dict[printer_i][np.argmin(nutility)][1] # min prediction
        
        
        sub_pred = prediction.filter(prediction.UserId == printer_i)
        regret_max = sub_pred.agg(F.min(sub_pred.columns[2])).collect()[0][0] # smallest GT value
        
        candidates.append((user,movie))
        if iter_i == 0:
            cumu_regrets_largest_colab[printer_i,iter_i] = abs(rating - regret_max)
            if abs(rating - regret_max)==0:
                first_find_minimal_tab[printer_i, 0] = min(iter_i, first_find_minimal_tab[printer_i, 0])
            
        else:
            cumu_regrets_largest_colab[printer_i,iter_i] = abs(rating - regret_max)
            if abs(rating - regret_max)==0:
                first_find_minimal_tab[printer_i, 0] = min(iter_i, first_find_minimal_tab[printer_i, 0])
            
            
    
    for user_i,movie_i in candidates:
        row_to_move = test.filter((test.UserId == user_i) & (test.MovieId == movie_i))
        train_updated = train.union(row_to_move)
        test_updated = test.filter(~((test.UserId == user_i) & (test.MovieId == movie_i)))
        train = train_updated
        test = test_updated





for printer_i in trange(row_size, desc='Printers'):
    train = spark.read.csv(path=PATH+'new_rating_data_train_y_'+str(printer_i+1)+'.csv',schema=schema)
    test = spark.read.csv(path=PATH+'new_rating_data_test_y_'+str(printer_i+1)+'.csv',schema=schema)
    train_random = spark.read.csv(path=PATH+'new_rating_data_train_y_'+str(printer_i+1)+'.csv',schema=schema)
    test_random = spark.read.csv(path=PATH+'new_rating_data_test_y_'+str(printer_i+1)+'.csv',schema=schema)
    print(printer_i,missing_N)
    for iter_i in range(missing_N):
        model = als_sub.fit(train)
        prediction = model.transform(test)
        
        max_value = prediction.agg(F.min(prediction.columns[3])).collect()[0][0]
        regret_max = prediction.agg(F.min(prediction.columns[2])).collect()[0][0]
        
        res = prediction.filter(prediction.prediction==max_value)
        
        
        res_ind = np.array(res.select(["UserId","MovieId","Rating","prediction"]).rdd.map(lambda row: (row[0],row[1],row[2],row[3])).collect())
        user,movie,rating,pred = res_ind[0]
        
        
        if iter_i == 0:
            cumu_regrets_largest_solo[printer_i, iter_i] = abs(rating - regret_max)
            if abs(rating - regret_max)==0:
                first_find_minimal_tab[printer_i, 2] = min(iter_i, first_find_minimal_tab[printer_i, 2]) # find it
            
        else:
            cumu_regrets_largest_solo[printer_i, iter_i] = abs(rating - regret_max)
            if abs(rating - regret_max)==0:
                first_find_minimal_tab[printer_i, 2] = min(iter_i, first_find_minimal_tab[printer_i, 2]) # find it
            

        row_to_move = test.filter((test.UserId == user) & (test.MovieId == movie))
        train_updated = train.union(row_to_move)
        test_updated = test.filter(~((test.UserId == user) & (test.MovieId == movie)))
        train = train_updated
        test = test_updated
        
        

print(first_find_minimal_tab)
np.savetxt(os.path.join(RESULTS_DIR,"first_find_minimal_tab.csv"),first_find_minimal_tab,delimiter=",")
np.savetxt(os.path.join(RESULTS_DIR,"cumu_regrets_largest_prediction.csv"),cumu_regrets_largest_colab, delimiter=",")
np.savetxt(os.path.join(RESULTS_DIR,"cumu_regrets_solo_largest_prediction.csv"),cumu_regrets_largest_solo, delimiter=",")


spark.stop()
