import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
def generate_features(t):
	t['date_time']=pd.to_datetime(t['date_time'])
	t['srch_ci'] = pd.to_datetime(t['srch_ci'])
	t['srch_co'] = pd.to_datetime(t['srch_co'])

	props = {}
	for prop in ["month", "dayofweek", "quarter"]:
		props[prop] = getattr(t["date_time"].dt, prop)
	date_props = ["month", "dayofweek", "quarter"]
	for prop in date_props:
		props["ci_{0}".format(prop)] = getattr(t["srch_ci"].dt, prop)
		props["co_{0}".format(prop)] = getattr(t["srch_co"].dt, prop)
	props["stay_span"] = (t["srch_co"] - t["srch_ci"]).astype('timedelta64[h]')
	props["hotel_cluster"] = t["hotel_cluster"]

	ret = pd.DataFrame(props)
	ret.fillna(-1, inplace=True)
	return ret
train = pd.read_csv('Data/small_train.csv')
train_ = generate_features(train)
predictors = [c for c in train_.columns if c not in ["hotel_cluster"]]

train_feature = train_[predictors]
train_label = train_["hotel_cluster"]

clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
scores = cross_validation.cross_val_score(clf, train_feature, train_label, cv=3)
print scores