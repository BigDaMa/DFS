import pickle
import numpy as np
import matplotlib.pyplot as plt
import webbrowser

from eli5.sklearn.explain_prediction import explain_prediction_linear_classifier
from eli5 import show_weights

pipeline = pickle.load(open("/tmp/pipeline.p", "rb"))
featurenames = pickle.load(open("/tmp/featurenames.p", "rb"))

clf = pipeline.named_steps['clf']

coefficients = clf.coef_[0]

assert len(featurenames) == len(coefficients)

print(coefficients)
print(featurenames)

ids = np.argsort((coefficients**2) * -1)

'''
fig, ax = plt.subplots()
plt.bar(range(len(featurenames)), coefficients[ids])
plt.xticks(range(len(featurenames)), np.array(featurenames)[ids], rotation='vertical')
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.5)
plt.show()
'''

'''
fig, ax = plt.subplots()

y_pos =range(len(featurenames))

ax.barh(y_pos, coefficients[ids],align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(np.array(featurenames)[ids])
ax.invert_yaxis()  # labels read top-to-bottom
#ax.set_xlabel('Performance')
#ax.set_title('How fast do you want to go today?')

plt.show()
'''

print(len(coefficients))
count_zero = len(coefficients) - np.count_nonzero(coefficients)
print(count_zero)

for i in range(len(featurenames)):
	if coefficients[i] == 0:
		print(featurenames[i])

#print(explain_prediction_linear_classifier(clf=clf, feature_names=featurenames))
htmlobj = show_weights(estimator=clf, feature_names=featurenames)

with open('/tmp/disppage.htm','w+') as f:   # Use some reasonable temp name
	f.write(htmlobj.data)

# open an HTML file on my own (Windows) computer
url = '/tmp/disppage.htm'
webbrowser.open(url)