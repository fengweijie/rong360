#coding:utf-8
from src.thundergbm_scikit import *
from sklearn.datasets import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from src.config_tool import read_config_ini

# x,y = load_svmlight_file("../lib/test_dataset.txt")
# clf = TGBMModel()
# clf.fit(x,y)
#
# x2,y2=load_svmlight_file("../lib/test_dataset.txt")
# y_predict=clf.predict(x2)
#
# rms = sqrt(mean_squared_error(y2, y_predict))
# print(rms)

model_params = read_config_ini("./config/model.ini")
raw_data_params = read_config_ini("./config/raw_data.ini")

df_train_feature_1_path = raw_data_params["cache_path"]["df_train_feature_1_path"]
data = np.load(df_train_feature_1_path)
X_train = data['X_train']
Y_train = data['y_train1']
feature_name = list(data["feature_name"])
model_param = model_params.get("lightgbm_binary_class",False)

model_name = model_params["file_path"]["model_name"]
test_size = model_params["file_path"]["test_size"]
output_path = model_params["file_path"]["output_path"]

thundergbm_params={
"n_device":1,
"min_child_weight":1.0,
"lambda_tgbm":1.0,
"gamma":1.0,
"max_num_bin":255,
"verbose":0,
"column_sampling_rate":1.0,
"bagging":0,
"n_parallel_trees":1,
"learning_rate":0.1,
"objective":"reg:logistic",
"num_class":1,
"out_model_name":"tgbm.model",
"in_model_name":"tgbm.model",
"tree_method":"auto"
}


clf = TGBMModel()

clf.fit(X_train,Y_train)

