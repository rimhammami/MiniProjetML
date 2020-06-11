%matplotlib inline
import pandas as pd
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# We can jump directly to working with the clean data because we saved our cleaned data set
data_clean = pd.read_csv('teamselectionTrain.csv')



all_inputs = data_clean[['Height', 'Weight']].values

all_labels = data_clean['Selection'].values

# This is the classifier that came out of Grid Search
k_neighors_classifier = KNeighborsClassifier(metric='euclidean', n_neighbors=28, weights= 'distance')

# All that's left to do now is plot the cross-validation scores
knn_classifier_scores = cross_val_score(k_neighors_classifier, all_inputs, all_labels, cv=10)
sb.boxplot(knn_classifier_scores)
sb.stripplot(knn_classifier_scores, jitter=True, color='black')

# ...and show some of the predictions from the classifier
(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.2)

k_neighors_classifier.fit(training_inputs, training_classes)

for input_features, prediction, actual in zip(testing_inputs[:10],
                                              k_neighors_classifier.predict(testing_inputs[:10]),
                                              testing_classes[:10]):
    print('{}\t-->\t{}\t(Actual: {})'.format(input_features, prediction, actual))