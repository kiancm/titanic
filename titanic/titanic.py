import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier

train = pd.read_csv('all/train.csv')
test = pd.read_csv('all/test.csv')

def clean(df):
    return (df.set_index('PassengerId')
             .drop(['Name', 'Ticket', 'Cabin'], axis=1)
             .pipe(lambda df: pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked']))
             .fillna(0))

train = clean(train)
test = clean(test)

# model = SGDClassifier(loss='log', verbose=3, penalty='none')
model = LogisticRegression(solver='liblinear', penalty='l1')
model.fit(train.drop('Survived', axis=1), train.Survived)
score = model.score(train.drop('Survived', axis=1), train.Survived)
preds = model.predict(test)
print(score)

test.assign(Survived=preds).Survived.to_frame().to_csv('results.csv')