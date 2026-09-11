import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('environmental_dataset.csv')

df['HeatRisk'] = 0.5*df['Temperature'] + 0.3*(100-df['Humidity']) + 0.2*df['WindSpeed']
df['PollutionRisk'] = 0.6*df['PM2.5'] + 0.4*df['PM10']
df['ClimateSeverity'] = 0.6*df['HeatRisk'] + 0.4*df['PollutionRisk']

def label(r):
    if r['ClimateSeverity'] > 60:
        return 'Mixed Strategy'
    if r['HeatRisk'] > 40 and r['PollutionRisk'] > 30:
        return 'Plant Trees + Air Pollution Control'
    if r['PollutionRisk'] > 40:
        return 'Air Pollution Control'
    if r['HeatRisk'] > 35:
        return 'Cool Roof'
    if r['HeatRisk'] > 25:
        return 'Green Roof'
    return 'No Immediate Action'

df['Recommendation'] = df.apply(label, axis=1)

X = df[['Temperature','Humidity','WindSpeed','PM2.5','PM10','HeatRisk','PollutionRisk','ClimateSeverity']]
y = df['Recommendation']

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(Xtr, ytr)
print('Accuracy:', accuracy_score(yte, model.predict(Xte)))
joblib.dump(model, 'model.pkl')
print('model.pkl saved')
