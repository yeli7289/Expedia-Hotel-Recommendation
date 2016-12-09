import pandas as pd
import numpy as np
import heapq
from collections import defaultdict
from MapScore import MapScore

# def generate_features(t):
# 	t['srch_ci'] = pd.to_datetime(t['srch_ci'])
# 	t['srch_co'] = pd.to_datetime(t['srch_co'])
# 	t['span'] = (t["srch_co"] - t["srch_ci"]).astype('timedelta64[h]')/24
# 	t.drop(['srch_ci','srch_co'])

def K_value(lA,lB):
	if len(lA)!=len(lB):
		raise Exception('two sequence with different length')
	count=0
	for i in range(len(lA)):
		if lA[i]==lB[i]:
			count+=1
	return float(count)/len(lA)

def Score(position, K):
	return pow(K,2)
def Five_predicted(dic):
	return [hc for hc, K in sorted(dic.items(), key=lambda x:x[1], reverse=True)[:5]]

train = pd.read_csv('Data/small_train.csv',dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_market':np.int32, 'hotel_cluster':np.int32, 'user_location_country':np.int32, 'user_location_region':np.int32,\
		'hotel_country':np.int32,'month':np.int32,'srch_adults_cnt':np.int32,'srch_children_cnt':np.int32},
	usecols=['srch_destination_id','is_booking','hotel_market','hotel_cluster','user_location_country','user_location_region','hotel_country','month','srch_ci','srch_co','srch_children_cnt','srch_adults_cnt'])

test = pd.read_csv('Data/small_test.csv',dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_market':np.int32, 'hotel_cluster':np.int32, 'user_location_country':np.int32, 'user_location_region':np.int32,\
		'hotel_country':np.int32,'month':np.int32,'srch_adults_cnt':np.int32,'srch_children_cnt':np.int32},
	usecols=['srch_destination_id','is_booking','hotel_market','hotel_cluster','user_location_country','user_location_region','hotel_country','month','srch_ci','srch_co','srch_children_cnt','srch_adults_cnt'])

MAP = MapScore()
dic_sdid = defaultdict(list)
train["srch_ci"] = pd.to_datetime(train['srch_ci'])
train["srch_co"] = pd.to_datetime(train['srch_co'])
train['span'] = (train["srch_co"] - train["srch_ci"]).astype('timedelta64[h]')/24
train = train.drop(['srch_ci','srch_co'],1)
test["srch_ci"] = pd.to_datetime(test['srch_ci'])
test["srch_co"] = pd.to_datetime(test['srch_co'])
test['span'] = (test["srch_co"] - test["srch_ci"]).astype('timedelta64[h]')/24
test = test.drop(['srch_ci','srch_co'],1)

most_pop = list(train["hotel_cluster"].value_counts()[:5].index)

# training procedure, clustering the data with srch_destination_id
for index, row in train.iterrows():
	dic_sdid[row["srch_destination_id"]].append(index)

# testing procedure
for index, row in test.iterrows():
	sdid = row["srch_destination_id"]
	nearest_element = []
	dic = defaultdict(int)
	for id in dic_sdid[sdid]:
		K = K_value(list(train.iloc[id].drop(['srch_destination_id','hotel_cluster'])), list(row.drop(['srch_destination_id','hotel_cluster'])))
		if len(nearest_element)<50:
			heapq.heappush(nearest_element,(K, train.iloc[id]['hotel_cluster']))
		else:
			if K>nearest_element[0]:
				heap.heappop()
				heap.heappush(nearest_element,(K,train.iloc[id]['hotel_cluster']))
	for i, (k,hc) in enumerate(nearest_element):
		dic[hc]+=Score(i,k)
	predict = Five_predicted(dic)
	if len(predict)==5:
		MAP.update_score(row['hotel_cluster'],predict)
	else:
		MAP.update_score(row['hotel_cluster'],predict+most_pop[:5-len(predict)])
print MAP.output_score()



