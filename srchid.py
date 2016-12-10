# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
from collections import defaultdict
from MapScore import MapScore

train = pd.read_csv('Data/small_train.csv',dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_market':np.int32, 'hotel_cluster':np.int32,'hotel_country':np.int32,'user_location_country':np.int32,'user_location_region':np.int32},\
	usecols=['srch_destination_id','is_booking','hotel_market','hotel_cluster','hotel_country','user_location_country','user_location_region'])
test = pd.read_csv('Data/small_test.csv',dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_market':np.int32, 'hotel_cluster':np.int32, 'hotel_country':np.int32,'user_location_country':np.int32,'user_location_region':np.int32},\
	usecols=['srch_destination_id','is_booking','hotel_market','hotel_cluster','hotel_country','user_location_country','user_location_region'])

MAP = MapScore()
weight = 0.05
target_feature = "user_location_region"

dic_srchid = defaultdict(lambda: defaultdict(int))
dic_srchid_result = defaultdict(list)
most_pop = list(train[target_feature].value_counts()[:5].index)

for index, row in train.iterrows():
	if row["is_booking"]:
		dic_srchid[row[target_feature]][row["hotel_cluster"]]+=1
	else:
		dic_srchid[row[target_feature]][row["hotel_cluster"]]+=weight

for key, value in dic_srchid.iteritems():
	n = len(value)
	if n>=5:
		dic_srchid_result[key] = [tup[0] for tup in sorted(value.iteritems(), key=lambda x:x[1], reverse=True)[:5]]
	else:
		dic_srchid_result[key] = [tup[0] for tup in sorted(value.iteritems(), key=lambda x:x[1], reverse=True)[:n]]+most_pop[:5-n]
print "finished encoding"

for index, row in test.iterrows():
	if row[target_feature] in dic_srchid_result:
		predict = dic_srchid_result[row[target_feature]]
	else:
		predict = most_pop
	MAP.update_score(row["hotel_cluster"], predict)

print MAP.output_score()

