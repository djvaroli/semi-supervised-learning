import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator


target_column_name = "Purchased?"
# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('../data/Next_month_products.csv', sep=',', dtype=np.float64)
features = tpot_data.drop(target_column_name, axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data[target_column_name], random_state=None)

# Average CV score on the training set was: 0.8754
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_classif, alpha=0.03),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.6000000000000001, min_samples_leaf=8, min_samples_split=18, n_estimators=100)),
    DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=1, min_samples_split=2)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
