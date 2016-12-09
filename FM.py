# Ranking method
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from fastFM import als
from numpy import array
import csv
from MapScore import MapScore
def OneHotGenerate(data):
	enc = OneHotEncoder()
	enc.fit(data)
	return enc


#preprocessing the catagorical data
train = pd.read_csv('Data/small_train.csv')
test = pd.read_csv('Data/small_test.csv')
catagory = ['site_name','posa_continent','user_location_country',\
'user_location_region','user_location_city','user_id','srch_destination_id',\
'srch_destination_type_id','is_booking','hotel_continent','hotel_country',\
'hotel_market','month']
tr_cat = train[catagory] 
enc = OneHotGenerate(tr_cat)

x_train = enc.transform(tr_cat)
tst_cat = test[catagory]
x_test = enc.transform(tst_cat)
trainer = []
print "finish encoding data"
# for every class, form a FM classfier and store it in trainer
for i in range(1,101):
	y_train=array([1 if train.iloc[j].hotel_cluster==i else 0 for j in range(len(train))])
	fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
	fm.fit(x_train, y_train)
	trainer.append(fm)
	print ("%d model trained" % i)
print "finish training FM model"
tst_matrix=[]
for tst_sample in x_test:
	tst_array=[]
	for i in range(len(trainer)):
		tst_array.append(trainer[i].predict(tst_sample)[0])
	tst_matrix.append(tst_array)
output = pd.DataFrame(tst_matrix)
output.to_csv('Data/FM_matrix.csv',index=False)

# target = pd.read_csv('Data/small_test.csv', usecols=['hotel_cluster'])
# target = target['hotel_cluster'].values.tolist()
# Map = MapScore()
# with open("Data/FM_matrix.csv", 'rb') as f:
# 	reader = csv.reader(f)
# 	for i, row in enumerate(reader):
# 		if i==0:
# 			continue
# 		row = map(float, row)
# 		predict = sorted(row, reverse=True)[:5]
# 		Map.update_score(target[i-1],[row.index(p) for p in predict])
# print Map.output_score()







