#coding:utf-8
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
from src.mylog import logger

def get_model_feature_importance(model,model_name="default",outputpath="./",importance_type='gain',num_feature=20):
    '''
    lightgbm 画importace函数
    :param model:
    :param model_name:
    :param outputpath:
    :param importance_type:
    :param num_feature:
    :return:
    '''
    try:
        outputpath = outputpath+model_name+"_feature_importance.png"
        ax = lgb.plot_importance(model,
                                 figsize=(20,13),
                                 importance_type= importance_type,
                                 title="gain_importance",
                                 max_num_features=num_feature)
        plt.savefig(outputpath)
    except:
        logger.error("create model feature fail.")
        return False
    else:
        logger.info("create model feature sucess.")
        return True


def get_model_train_result(evals_result, model_name="default", outputpath="./"):
    '''
    画出训练结果函数
    :param evals_result:
    :param model_name:
    :param outputpath:
    :return:
    '''
    try:
        outputpath = outputpath+model_name+"_train_result.png"
        ax = lgb.plot_metric(evals_result,metric='binary_logloss',
                                 figsize=(20,13)
                                 )
        plt.savefig(outputpath)
    except:
        logger.error("create model train result fail.")
        #raise RuntimeError("create model train result fail.")
        return False
    else:
        logger.info("create model train result sucess.")
        return True


def get_model_tree_visual(model,model_name="default",tree_index=1,outputpath="./"):
    '''

    :param model:
    :param model_name:
    :param outputpath:
    :param importance_type:
    :param num_feature:
    :return:
    '''
    try:
        outputpath = outputpath+model_name+"_tree.png"
        ax = lgb.plot_tree(model,
                           tree_index=tree_index,
                           figsize=(20,13),
                           )
        plt.savefig(outputpath)
    except:
        logger.error("create model tree fail.")
        return False
    else:
        logger.info("create model tree sucess.")
        return True

def get_model_tree_digraph(model,model_name="default",outputpath="./"):
    '''
    画出tree 的树结构
    :param model:
    :param model_name:
    :param outputpath:
    :param importance_type:
    :param num_feature:
    :return:
    '''
    try:
        outputpath = outputpath+model_name+"_tree_digraph.gv"
        graph = lgb.create_tree_digraph(model,name=model_name)
        graph.render(filename=outputpath)
    except:
        logger.error("create model tree_digrap fail.")
        return False
    else:
        logger.info("create model tree_digrap sucess.")
        return True


def save_model_as_text(model,model_name="default",outputpath="./"):
    '''
    保存模型
    :param model:
    :param model_name:
    :param outputpath:
    :return:
    '''
    try:
        outputpath = outputpath + model_name + "_lightgbm.txt"
        model.save_model(outputpath)
    except:
        logger.error("save lightgbm model fail.")
        return False
    else:
        logger.info("save lightgbm model sucess")
        return True


def load_model_from_text(model_file):
    '''
    加载模型
    :param model_file:
    :return:
    '''
    try:
        bst = lgb.Booster(model_file=model_file)
    except:
        logger.error("load model fail.")
        return False
    else:
        logger.info("load model from text sucess")
        return bst


def train_lightgbm_model(X_train, Y_train, feature_name, model_param,test_size=0.2):
    '''
    训练lightgbm 给定参数x,y,x所属特征,分裂的结果,model训练的参数
    :param X_train:
    :param Y_train:
    :param feature_name:
    :param test_size:
    :param model_param:
    :return:
    '''
    try:
        x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=test_size, random_state=42)
        data_train = lgb.Dataset(x_train, y_train, feature_name=feature_name)
        data_test = lgb.Dataset(x_test, y_test, feature_name=feature_name)
        evals_result = {}
        gbm = lgb.train(model_param, data_train, num_boost_round=10000,
                        evals_result=evals_result,
                        valid_sets=[data_test], early_stopping_rounds=50)

        y_train_pred = gbm.predict(x_train)
        train_acc = roc_auc_score(y_train, y_train_pred)
        y_test_pred =gbm.predict(x_test)
        test_acc = roc_auc_score(y_test, y_test_pred)

        logger.info("lightgbm model train auc:" + str(train_acc) + " test auc:" + str(test_acc))
        return gbm,evals_result
    except:
        logger.error("light/gbm model train fail.")
        return False,False



def train_binaryclass_baseline_model(model_params,raw_data_params):
    df_train_feature_1_path = raw_data_params["cache_path"]["df_train_feature_1_path"]
    data = np.load(df_train_feature_1_path)
    X_train = data['X_train']
    Y_train = data['y_train1']
    feature_name = list(data["feature_name"])
    model_param = model_params.get("lightgbm_binary_class",False)

    model_name = model_params["file_path"]["model_name"]
    test_size = model_params["file_path"]["test_size"]
    output_path = model_params["file_path"]["output_path"]

    gbm,evals_result= train_lightgbm_model(X_train, Y_train, feature_name, model_param,test_size=test_size)
    if gbm and evals_result:
        get_model_feature_importance(gbm,model_name=model_name,outputpath=output_path)
        print(evals_result)
        get_model_train_result(evals_result,model_name=model_name,outputpath=output_path)
        get_model_tree_digraph(gbm,model_name=model_name,outputpath=output_path)
        save_model_as_text(gbm,model_name=model_name,outputpath=output_path)
        logger.info("baseline train fianlly sucess.")



