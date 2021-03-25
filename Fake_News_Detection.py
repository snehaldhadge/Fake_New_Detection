import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("news.csv")

print(data.head())

print(data.shape)

label = data.label

print(label)

plt.figure(figsize=(7, 7))
sns.set(style="darkgrid")

color = sns.color_palette("Set1")
ax = sns.countplot(x="label", data=data, palette=color)

ax.set(xticklabels=['FAKE', 'REAL'])

plt.title("Data distribution of fake and real data")

X_train, X_test, y_train, y_test = train_test_split(data['text'],label, test_size=0.2, random_state=42)

# Initializing the TfidVectorizer with English stop words and
#  maximum document frequency of 0.7 (terms with a higher document frequency will be discarded)

#  ( "tdif_vect", TfidfVectorizer(stop_words="english",max_df=0.7)),
#    ("pac", PassiveAggressiveClassifier(max_iter=50))
#])
#pipeline.fit(X_train,y_train)

tdif_vect= TfidfVectorizer(stop_words="english",max_df=0.7)
tfidf_train = tdif_vect.fit_transform(X_train)
tfidf_test = tdif_vect.transform(X_test)

# PAC algorithm remains passive for a correct classification
# outcome, and turns aggressive in the event of a miscalculation,
# updating and adjusting

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Testing for Accuracy
y_pred = pac.predict(tfidf_test)

print("Accuracy Score: ",round(accuracy_score(y_pred,y_test)* 100,2))

# The Accuracy is 93%