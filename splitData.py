# separate training file into smaller training and testing set for development
import random
import pandas as pd

train = pd.read_csv("Data/train.csv")

user_id = train.user_id.unique()
# select 10000 users for the model evaluation purpose
sel_user_id = [user_id[i] for i in sorted(random.sample(range(len(user_id)), 10000)) ]
sel_train = train[train.user_id.isin(sel_user_id)]


sel_train["date_time"] = pd.to_datetime(sel_train["date_time"])
sel_train["year"] = sel_train["date_time"].dt.year
sel_train["month"] = sel_train["date_time"].dt.month
for m in range(1,13):
	train_month = sel_train[sel_train.month == m]
	if m==1:
		new_train = train_month.sample(frac=0.8)
		new_test = train_month.loc[~train_month.index.isin(new_train.index)]
	else:
		temp_train = train_month.sample(frac=0.8)
		temp_test = train_month.loc[~train_month.index.isin(temp_train.index)]
		new_train = new_train.append(temp_train)
		new_test = new_test.append(temp_test)

new_test = new_test[new_test.is_booking == True]
new_train.to_csv('Data/small_train.csv')
new_test.to_csv('Data/small_test.csv')
