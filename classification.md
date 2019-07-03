Pulsar classification is a great example of where machine learning can be used beneficially in astrophysics. It's not the most straightforward classification problem, but here I'm going to outline the basics using the scikit-learn random forest classifier. This post was inspired by <a href="http://www.scienceguyrob.com/">Rob Lyon</a>'s pulsar classification tutorials in the <a href="https://github.com/astro4dev/OAD-Data-Science-Toolkit/tree/master/Teaching%20Materials/Machine%20Learning/Supervised%20Learning/Examples/PPC">IAU OAD Data Science Toolkit</a>.
<div style="border:1px solid red;padding:16px;text-align:center;">This post is in Python 3.</div>
<h3>I see dead... stars?</h3>
Pulsars are "pulsating radio sources", now known to be caused by rapidly rotating neutron stars. <a href="https://en.wikipedia.org/wiki/Neutron_star" target="_blank" rel="noopener noreferrer">Neutron stars</a> are the relics of dead massive stars, they're small and extremely dense - think about something the same mass as the Sun crammed into a radius roughly the same as the <a href="https://en.wikipedia.org/wiki/M25_motorway" target="_blank" rel="noopener noreferrer">M25 motorway</a>. You can read all about them <a href="http://www.jb.man.ac.uk/distance/frontiers/pulsars/section1.html" target="_blank" rel="noopener noreferrer">here</a>.

[caption id="attachment_3537" align="aligncenter" width="320"]<img class="alignnone size-full wp-image-3537" src="https://allofyourbases.files.wordpress.com/2019/03/lightnew.gif" alt="lightnew" width="320" height="240" /> Enter aAn <a href="http://www.astron.nl/pulsars/animations/">artist's impression of a pulsar</a>. Image credit: <a href="http://www.astron.nl/astronomy-group/people/joeri-van-leeuwen/joeri-van-leeuwen">Joeri van Leeuwen</a>, License: <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC-BY-AS</a> caption[/caption]

You can even <a href="http://www.jb.man.ac.uk/distance/frontiers/pulsars/section1.html" target="_blank" rel="noopener noreferrer">listen to them</a> (if you really want to...)

[audio src="https://allofyourbases.files.wordpress.com/2019/03/b0329.wav"][/audio]

<b>PSR B0329+54</b>: This pulsar is a typical, normal pulsar, rotating with a period of 0.714519 seconds, i.e. close to 1.40 rotations/sec.

Pulsars are pretty interesting objects in their own right, they are used as a probe of stellar evolution as well as being used <a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwiG5PXzrvXgAhXHXRUIHZeICTYQFjACegQIAxAB&url=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS1387647304000909&usg=AOvVaw1IKp5W-f3gu3QCjmDCpBYR" target="_blank" rel="noopener noreferrer">to test general relativity</a> due to their extremely high densities. These days they're also used to detect and map <a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwjWt9XfrvXgAhXdTBUIHZUDCA8QFjACegQIARAB&url=https%3A%2F%2Fwww.skatelescope.org%2Fnewsandmedia%2Foutreachandeducation%2Fskawow%2Fgravitational-wave%2F&usg=AOvVaw2tEsK4ezAUVVM4K62KVnnW" target="_blank" rel="noopener noreferrer">gravitational wave signatures</a>. However, identifying them in the data streams from radio telescopes is not trivial. There are lots of man-made sources of radio frequency interference that can mimic the signals from pulsars. Classifying candidate data samples as <em>pulsar</em> or <em>not pulsar</em> is serious business.

The individual pulses are all different, so astronomers stack them up and create an average integrated pulse profile to characterise a particular pulsar:

[caption id="attachment_3543" align="aligncenter" width="302"]<img class="  wp-image-3543 aligncenter" src="https://allofyourbases.files.wordpress.com/2019/03/pulsestack-e1552071632736.gif" alt="pulsestack" width="302" height="477" /> <a href="https://www.cv.nrao.edu/~sransom/web/Ch6.html" target="_blank" rel="noopener noreferrer">Essentials of Radio Astronomy</a>[/caption]

Additionally the pulse will arrive at different times across different radio frequencies. The delay from frequency to frequency is caused by the ionised inter-stellar medium and is known as the <em><strong>dispersion</strong></em>. It looks like this:

[caption id="attachment_3544" align="aligncenter" width="413"]<img class="  wp-image-3544 aligncenter" src="https://allofyourbases.files.wordpress.com/2019/03/dispersion.png" alt="dispersion" width="413" height="525" /> <a href="https://www.cv.nrao.edu/~sransom/web/Ch6.html" target="_blank" rel="noopener noreferrer">Essentials of Radio Astronomy</a>[/caption]

Astronomers fit for the shape of the delay in order to compensate for its effect, but there's always an uncertainty associated with the fit. That is expressed in the DM-SNR ("dispersion-measure-signal-to-noise-ratio") curve, which looks like this:

<img class=" size-full wp-image-3545 aligncenter" src="https://allofyourbases.files.wordpress.com/2019/03/dm_snr.png" alt="dm_snr" width="368" height="219" />

When you put these two curves together it means that for each pulsar candidate there are <strong>eight numerical features</strong> that can be extracted as standard: four from the integrated pulse profile and four from the DM-SNR curve:

<img class="aligncenter size-full wp-image-3538" src="https://allofyourbases.files.wordpress.com/2019/03/pulsar_features.png?w=1052" alt="pulsar_features.png" width="526" height="352" />
<h3></h3>
<h3>Getting set-up</h3>
First some general libraries:

[code language="python" gutter="false" highlight="1-100"]
import numpy as np   # for array stuff
import pylab as pl   # for plotting stuff
import pandas as pd  # for data handling
[/code]

Then a bunch of scikit-learn libraries:

[code language="python" gutter="false" highlight="1-100"]
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
[/code]

I'm also using <a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjZmfDhqfXgAhV1unEKHU73BRsQFjAAegQIBRAC&url=https%3A%2F%2Fscikit-plot.readthedocs.io%2Fen%2Fstable%2F&usg=AOvVaw1BmgIU7oy2JWfM6QRyjnZw" target="_blank" rel="noopener noreferrer">scikit-plot</a>, which I only recently discovered and has made my life much easier :-)

[code language="python" gutter="false" highlight="1-100"]
import scikitplot as skplt
[/code]

I'm using the <a href="https://archive.ics.uci.edu/ml/datasets/HTRU2" target="_blank" rel="noopener noreferrer">HTRU2 dataset</a>. This dataset compiles the eight features described above for both 1,639 true known pulsars, as well as 16,259 additional candidate pulsars later identified to be RFI/noise. You can find a full description of the dataset in <a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiKj-aKqfXgAhX1sHEKHeMdAIYQFjAAegQIARAB&url=https%3A%2F%2Farxiv.org%2Fabs%2F1603.05166&usg=AOvVaw1K0x75Q0pLRZqpvTbd5nMD" target="_blank" rel="noopener noreferrer">this paper</a>.

I added a row to the CSV for the feature names for the purpose of this example - you can find my version in the IAU OAD Data Science Toolkit <a href="https://github.com/astro4dev/OAD-Data-Science-Toolkit/blob/master/Teaching%20Materials/Machine%20Learning/Supervised%20Learning/Examples/PPC/Data/pulsar.csv" target="_blank" rel="noopener noreferrer">here</a>.

[code language="python" gutter="false" highlight="1-100"]
df = pd.read_csv('data/pulsar.csv')
[/code]

You can take a look at the names of the features in the file like this (<code>pf</code> = integrated profile & <code>dm</code> = DM-SNR curve):

[code language="python" gutter="false" highlight="1-100"]
feature_names = df.columns.values[0:-1]
print(feature_names)
[/code]

<code>['mean_int_pf' 'std_pf' 'ex_kurt_pf' 'skew_pf' 'mean_dm' 'std_dm'
'kurt_dm' 'skew_dm']</code>

and we can check just how much data we're dealing with:

[code language="python" gutter="false" highlight="1-100"]
print ('Dataset has %d rows and %d columns including features and labels'%(df.shape[0],df.shape[1]))
[/code]

<code>Dataset has 17898 rows and 9 columns including features and labels</code>

We're going to start by separating the numerical feature data from the class labels for all the candidates. To get the feature data on its own we can just strip off the column containing the class labels:

[code language="python" gutter="false" highlight="1-100"]
features = df.drop('class', axis=1)
[/code]

The labels for each object tell us abut the target class and we can create an array of those data by extracting the column from the original dataset:

[code language="python" gutter="false" highlight="1-100"]
targets = df['class']
[/code]
<h3>Setting up the Machine Learning</h3>
Now we need to split our labelled data into two separate datasets: one to train the classifier and one to test the fitted machine learning model. To do this we can use the function <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank" rel="noopener noreferrer">train_test_split</a> from the <a href="https://scikit-learn.org/" target="_blank" rel="noopener noreferrer">scikit_learn</a> library:

[code language="python" gutter="false" highlight="1-100"]
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.33, random_state=66)
[/code]

At this point we now have our dataset in a suitable state to start training the classifier.

To start with we need to initiate the random forest classifier from <a href="https://scikit-learn.org/" target="_blank" rel="noopener noreferrer">scikit_learn</a>:

[code language="python" gutter="false" highlight="1-100"]
RFC = RandomForestClassifier(n_jobs=2,n_estimators=10)
[/code]

...and we can immediately fit the machine learning model to our training data:

[code language="python" gutter="false" highlight="1-100"]
RFC.fit(X_train,y_train)
[/code]

<code>RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
max_depth=None, max_features='auto', max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=2,
oob_score=False, random_state=None, verbose=0,
warm_start=False)</code>

We can then used the trained classifier to predict the label for the test data that we split out earlier:

[code language="python" gutter="false" highlight="1-100"]
rfc_predict = RFC.predict(X_test)
[/code]
<h3>Evaluating Performance</h3>
So how did we do? We need to evaluate the performance of our classifier.
A good first step is to evaluate the <a href="https://www.openml.org/a/estimation-procedures/1" target="_blank" rel="noopener noreferrer">cross-validation</a>. This will tell us how well our machine learning model generalises, i.e. whether we have over-fitted the training data.

[code language="python" gutter="false" highlight="1-100"]
rfc_cv_score = cross_val_score(RFC, features, targets, cv=10, scoring='roc_auc')
[/code]

Let's print out the various evaluation criteria:

[code language="python" gutter="false" highlight="1-100"]
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict, target_names=['Non Pulsar','Pulsar']))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
[/code]

<code>=== Confusion Matrix ===
[[5327   35]
[  93  452]]

=== Classification Report ===
precision recall f1-score support

Non Pulsar 0.98 0.99 0.99 5362
Pulsar 0.93 0.83 0.88 545

micro avg 0.98 0.98 0.98 5907
macro avg 0.96 0.91 0.93 5907
weighted avg 0.98 0.98 0.98 5907

=== All AUC Scores ===
[0.92774615 0.94807886 0.96225025 0.96079711 0.96652717 0.9472501
0.96336963 0.95761145 0.96597591 0.96716753]

=== Mean AUC Score ===
Mean AUC Score - Random Forest: 0.956677415292086</code>

We can make a more visual representation of the confusion matrix using the scikit-plot library. To do this we need to know the predictions from our cross validation, rather than the <a href="https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc" target="_blank" rel="noopener noreferrer">Area Under Curve (AUC)</a> value:

[code language="python" gutter="false" highlight="1-100"]
predictions = cross_val_predict(RFC, features, targets, cv=2)
[/code]

[code language="python" gutter="false" highlight="1-100"]
skplt.metrics.plot_confusion_matrix(targets, predictions, normalize=True)
[/code]

<img class=" size-full wp-image-3542 aligncenter" src="https://allofyourbases.files.wordpress.com/2019/03/conf_mat.png" alt="conf_mat" width="309" height="278" />

To plot the <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic" target="_blank" rel="noopener noreferrer">ROC curve</a> we need to find the probabilities for each target class separately. We can do this with the predict_proba function:

[code language="python" gutter="false" highlight="1-100"]
probas = RFC.predict_proba(X_test)
[/code]

[code language="python" gutter="false" highlight="1-100"]
skplt.metrics.plot_roc(y_test, probas)
[/code]

In a balanced data set there should be no difference between the micro-average ROC curve and the macro-average ROC curve. In the case where there is a class imbalance (like here), if the macro ROC curve is lower than the micro-ROC curve then there are more cases of mis-classification in minority class.

<img class=" size-full wp-image-3541 aligncenter" src="https://allofyourbases.files.wordpress.com/2019/03/roc.png" alt="roc.png" width="394" height="278" />

We can use the output of the <code>RFC.predict_proba( )</code> function to plot a <a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwiK0_nWsvXgAhUiTxUIHQh6B2YQFjACegQIBhAK&url=http%3A%2F%2Fscikit-learn.org%2Fstable%2Fauto_examples%2Fmodel_selection%2Fplot_precision_recall.html&usg=AOvVaw1UG9QgvJBEERHSf61yVi9K" target="_blank" rel="noopener noreferrer">Precision-Recall Curve</a>.

[code language="python" gutter="false" highlight="1-100"]
skplt.metrics.plot_precision_recall(y_test, probas)
[/code]

<img class=" size-full wp-image-3540 aligncenter" src="https://allofyourbases.files.wordpress.com/2019/03/precision_recall.png" alt="precision_recall.png" width="394" height="278" />
<h3>Ranking the Features</h3>
Let's take a look at the relative importance of the different features that we fed to our classifier:

[code language="python" gutter="false" highlight="1-100"]
importances = RFC.feature_importances_
indices = np.argsort(importances)
[/code]

[code language="python" gutter="false" highlight="1-100"]
pl.figure(1)
pl.title('Feature Importances')
pl.barh(range(len(indices)), importances[indices], color='b', align='center')
pl.yticks(range(len(indices)), feature_names[indices])
pl.xlabel('Relative Importance')

pl.show()
[/code]

<img class=" size-full wp-image-3539 aligncenter" src="https://allofyourbases.files.wordpress.com/2019/03/importances.png" alt="importances.png" width="420" height="278" />
