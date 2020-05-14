import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import tree


def one_hot_encoding(df, categorical_cols):
    # any non numeric column (with data type = object) is considered categorical
    for col in categorical_cols:
        one_hot_encoder = OneHotEncoder(sparse=False)
        col_onehot_encoded = one_hot_encoder.fit_transform(df[col].values.reshape(-1, 1))
        col_df_colnames = [col+name[2:] for name in one_hot_encoder.get_feature_names()]
        col_onehot_encoded_df = pd.DataFrame(data=col_onehot_encoded, columns=col_df_colnames)
        df = pd.concat([df, col_onehot_encoded_df], axis=1)
    return df


loans = pd.read_csv('data/lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop(['bad_loans'], axis=1)

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

loans = loans[features + [target]]
categorical_cols = [xcol for xcol in loans.columns if loans.dtypes[xcol] == 'object']
loans = one_hot_encoding(loans, categorical_cols)

safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print("Number of safe loans  : %s" % len(safe_loans_raw))
print("Number of risky loans : %s" % len(risky_loans_raw))

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
sampled_safe_loans_count = int(percentage * len(safe_loans_raw))
print('Sample safe loans count: %s' % sampled_safe_loans_count)
risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(sampled_safe_loans_count, random_state=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

print("Percentage of safe loans                 : %s" % (len(safe_loans) / float(len(loans_data))))
print("Percentage of risky loans                : %s" % (len(risky_loans) / float(len(loans_data))))
print("Total number of loans in our new dataset : %s" % len(loans_data))

# drop the categorical columns
loans_data = loans_data.drop(categorical_cols, axis=1)
X = loans_data.loc[:, loans_data.columns != target].values
y1d = loans_data[target].values
y = y1d.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decision_tree_model = tree.DecisionTreeClassifier(max_depth=2)
decision_tree_model.fit(X_train, y_train)
tree.plot_tree(decision_tree_model, feature_names=features, class_names=['safe', 'risky'], filled=True)
print('done')