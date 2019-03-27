#coding:utf-8
import pandas as pd
import numpy as np

from src.mylog import logger
from src.config_tool import  read_config_ini
from src.model_tool import train_binaryclass_baseline_model

model_params = read_config_ini("./config/model.ini")
raw_data_params = read_config_ini("./config/raw_data.ini")
train_binaryclass_baseline_model(model_params,raw_data_params)





