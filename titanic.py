import os

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier

train = pd.read_csv('all/train.csv')
test = pd.read_csv('all/test.csv')


def clean(df):
    return (
        df.set_index('PassengerId')
        .assign(
            complex_ticket=lambda df: df['Ticket']
            .str.count(' ')
            .astype(bool)
            .astype(int)
        )
        .assign(
            num_cabins=lambda df: df['Cabin']
            .str.count(' ')
            .apply(lambda x: x + 1)
            .fillna(0)
        )
        .assign(has_age=lambda df: df['Age'].isnull())
        .pipe(
            lambda df: pd.get_dummies(
                df, columns=['Sex', 'Pclass', 'Embarked', 'has_age']
            )
        )
        .fillna(0)
        .drop(['Name', 'Ticket', 'Cabin'], axis=1)
    )


train = clean(train)
test = clean(test)

# model = SGDClassifier(loss='log', verbose=3, penalty='none')
# model = LogisticRegression(solver='liblinear')
model = XGBClassifier()
print(model)
model.fit(train.drop('Survived', axis=1), train.Survived)

score = model.score(train.drop('Survived', axis=1), train.Survived)
preds = model.predict(test)
survived = sum(preds)
print(f'score: {score}')
print(f'survived: {survived}')
print(f'died: {len(preds) - survived}')

test.assign(Survived=preds).Survived.to_frame().to_csv('results.csv')
