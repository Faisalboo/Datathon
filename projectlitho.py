import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/content/CSV_train.csv", sep=';')
df = df[['WELL', 'DEPTH_MD', 'RDEP', 'RHOB', 'GR', 'DTC', 'FORCE_2020_LITHOFACIES_LITHOLOGY']]
df = df.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY': 'LITH'})
df.dropna(inplace=True)

# inputs and target
X = df[['RDEP', 'RHOB', 'GR', 'DTC']]
y = df['LITH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_test, y_pred)

cf_matrix = confusion_matrix(y_test, y_pred)

labels = ['Sandstone(30000)', 'Shale(65000)', 'Sandstone/Shale(65030)', 'Limestone(70000)', 'Chalk(70032)', 'Dolomite(74000)', 
            'Marl(80000)', 'Anhydrite(86000)', 'Halite(88000)','Coal(90000)', 'Basement(93000)', 'Tuff(99000)']   
fig = plt.figure(figsize=(11,11))
ax = sns.heatmap(cf_matrix, annot=True, cmap='crest', fmt='.1f', xticklabels=labels, yticklabels = labels)
ax.set_title('Confusion Matrix with labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')