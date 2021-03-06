# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
import math
import os
import re

pooling_nums = 8

def read_pro_data():
	pro_data = np.zeros((1, 11))
	pro_path =  r"C:\Users\Jh\Desktop\智慧中国杯\pro_mol_data\feature_pro"
	for filename in os.listdir(pro_path):
		fullname = os.path.join(pro_path, filename)
		with open(fullname, 'r') as f:
			pro_data_id = pd.read_csv(f)
		pro_data_id = pro_data_id.drop('Unnamed: 0', axis = 1)
		pro_id = int(re.sub("\D", "", filename)[:-1])
		# print (filename, pro_id)
		pro_data_id = np.transpose(pro_data_id)
		pro_data_id['Protein_ID'] = pro_id

		pooling_size = math.ceil(pro_data_id.shape[0] / pooling_nums)
		pooling_step = math.floor(pro_data_id.shape[0] / pooling_nums)
		cur_idx = 0
		for i in range(pooling_nums):
			p = pro_data_id[cur_idx : cur_idx + pooling_size]
			name = p.columns.values.tolist()
			p = pd.DataFrame(p[name].agg('mean'))
			p = pd.DataFrame(np.transpose(p))
			cur_idx = cur_idx + pooling_step - 1
			# print (pro_data.shape, p.shape)
			pro_data = np.row_stack((pro_data, p)) 

	pro_data = np.delete(pro_data, 0, 0)
	pro_data = pd.DataFrame(pro_data)
	pro_data.rename(columns = {10: 'Protein_ID'}, inplace = True)
	print (pro_data['Protein_ID'])
	pro_data[['Protein_ID']] = pro_data[['Protein_ID']].astype(int)
	return pro_data

def read_mol_data():
	mol_path = r"C:\Users\Jh\Desktop\智慧中国杯\pro_mol_data\feature_mol\feature_mol_NA2zero.csv"
	with open(mol_path, 'r') as f:
		mol_data = pd.read_csv(f)
	mol_data['Molecule_ID'] = [i for i in range(len(mol_data.index))]
	return mol_data

def read_Ki_data(dtype):
	if dtype == "train":
		Ki_path = r"C:\Users\Jh\Desktop\智慧中国杯\pro_mol_data\df_affinity_train.csv"
	else:
		Ki_path = r"C:\Users\Jh\Desktop\智慧中国杯\pro_mol_data\df_affinity_test_toBePredicted.csv"
	with open(Ki_path, 'r') as f:
		Ki_data = pd.read_csv(f)
	return Ki_data
 
if __name__ == "__main__":
	protein_data = read_pro_data()
	molecule_data = read_mol_data()
	
	train_data = read_Ki_data("train")
	test_data = read_Ki_data("test")
	test_data['Ki'] = -11
	Ki_data  =  pd.concat([train_data,test_data])

	# pro_data.to_csv("pro_data_lgb_pooling.csv",index=True,sep=',')

	data = pd.merge(Ki_data, protein_data, on = 'Protein_ID', how = 'left')
	data = pd.merge(data, molecule_data, on = 'Molecule_ID',  how = 'left')

	# data.to_csv("data_lgb_pooling.csv", index = True, sep = ',')
	print (data.shape)

	print ("Begin Training...")

	train_feat = data[data['Ki']> -11].fillna(0)
	test_feat = data[data['Ki']<=-11].fillna(0)
	label_x  = train_feat['Ki']
	label_y  = test_feat['Ki']

	submission = test_feat[['Protein_ID','Molecule_ID']]
	train_feat = train_feat.drop('Ki',axis=1)
	test_feat = test_feat.drop('Ki',axis=1)
	train_feat = train_feat.drop('Protein_ID',axis=1)
	test_feat = test_feat.drop('Protein_ID',axis=1)
	train_feat = train_feat.drop('Molecule_ID',axis=1)
	test_feat = test_feat.drop('Molecule_ID',axis=1)

	train = lgb.Dataset(train_feat, label=label_x)
	test  = lgb.Dataset(test_feat, label=label_y,reference=train)

	params = {
	    'boosting_type': 'gbdt',
	    'objective': 'regression_l2',
	    'metric': 'l2',
	    #'objective': 'multiclass',
	    #'metric': 'multi_error',
	    #'num_class':5,
	    'min_child_weight': 3,
	    'num_leaves': 2 ** 5,
	    'lambda_l2': 10,
	    'subsample': 0.7,
	    'colsample_bytree': 0.7,
	    'colsample_bylevel': 0.7,
	    'learning_rate': 0.05,
	    'tree_method': 'exact',
	    'seed': 2017,
	    'nthread': 12,
	    'silent': True
	}

	num_round = 3000
	gbm = lgb.train(params, train, num_round, verbose_eval=50, valid_sets=[train,test])
	preds_sub = gbm.predict(test_feat)

	name='lgb_pooling.csv'
	submission['Ki'] = preds_sub
	submission = pd.DataFrame(submission.groupby(['Protein_ID', 'Molecule_ID'])['Ki'].agg('mean')).reset_index()
	submission.to_csv(name, index=False)
