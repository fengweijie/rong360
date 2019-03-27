#coding:utf-8
import pandas as pd
import numpy as np

from src.mylog import logger
from src.config_tool import  read_config_ini

raw_data_params = read_config_ini("./config/raw_data.ini")
logger.info(raw_data_params)

df_user_info_train_path = raw_data_params["file_path"]["df_user_info_train_path"]
df_overdue_train_path = raw_data_params["file_path"]["df_overdue_train_path"]
df_train_feature_1_path = raw_data_params["cache_path"]["df_train_feature_1_path"]
df_user_info_train = pd.read_csv(df_user_info_train_path,header=None,names=["用户id","性别","职业","教育程度","婚姻状态",
                                                                            "户口类型"])
df_overdue_train = pd.read_csv(df_overdue_train_path,header=None,names=["用户id","样本标签"])
df_train = pd.merge(df_user_info_train,df_overdue_train,on="用户id",how="inner")
target_lable = list(df_train["样本标签"].copy())
df_train = df_train.drop(["用户id","样本标签"],axis = 1)
print(df_train[:10])

np.savez(df_train_feature_1_path,
         feature_name=df_train.columns,
         X_train= df_train,
         y_train1=target_lable
        )