#
# #训练放款时间表
# df_loan_time_train= pd.read_csv("../raw_data/train/loan_time_train.txt",
#                                 header=None,names=['用户标识','放款时间'])
# df_loan_time_train["放款时间"]=df_loan_time_train["放款时间"]/86400
#
# #训练用户表
# df_user_info_train = pd.read_csv("../raw_data/train/user_info_train.txt",header=None,
#                      names=['用户标识','用户性别','用户职业','用户教育程度','用户婚姻状态', '用户户口类型'])
# #训练信用卡账单表
# df_bill_detail_train =pd.read_csv("../raw_data/train/bill_detail_train.txt",header=None,
#                      names=['用户标识','时间','银行标识','上期账单金额','上期还款金额','信用卡额度',
#                            '本期账单余额','本期账单最低还款额','消费笔数','本期账单金额','调整金额',
#                           '循环利息','可用余额','预借现金额度','还款状态'])
# df_bill_detail_train["时间"] = df_bill_detail_train["时间"]/86400
# df_bill_detail_train = pd.merge(df_bill_detail_train,df_loan_time_train,how='inner',on="用户标识")
#
# #训练表
# df_overdue_train = pd.read_csv("../raw_data/train/overdue_train.txt",header=None,
#                      names=['用户标识','标签'])
# df_overdue_train = pd.merge(df_overdue_train,df_user_info_train,on="用户标识",how="inner")
# df_overdue_train = pd.merge(df_overdue_train,df_bill_detail_train,how='inner',on="用户标识")
#
# print(df_overdue_train.shape)
