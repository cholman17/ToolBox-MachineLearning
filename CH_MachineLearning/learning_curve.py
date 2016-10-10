""" Exploring learning curves for classification of handwritten digits """
from __future__ import division
import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


data = load_digits()
print data.DESCR
num_trials = 10
train_percentages = range(5,95,5) #FLAT LINE?! Train accuracy is terrible!
test_accuracies = numpy.zeros(len(train_percentages))

# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability --> AND save the last value?? Or average?

# TODO:
lst = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]  #Made separate list too; also getting flat line
alist= []
blist= []
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
for val in lst: 
#for val in train_percentages:
	l =  0
	h = []
	k = []
	for _ in range(num_trials):
		X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=(val/100))
		model = LogisticRegression(C=10**-10)
		model.fit(X_train, y_train)
		l+=1
		h.append(model.score(X_train,y_train))
		k.append(model.score(X_test,y_test))
		if l == num_trials:
			print [model.score(X_train,y_train),model.score(X_test,y_test)]
			print '+++++ MEAN VAL +++++'
			print [mean(h),mean(k)]
	print "Train accuracy %f for %s" %(model.score(X_train,y_train), val)
	print "Test accuracy %f" %model.score(X_test,y_test)
	alist.append(mean(h))
	blist.append(mean(k))

print alist
print blist
#ACTUAL TRAINING VALUES
fig = plt.gcf()
#plt.plot(train_percentages, test_accuracies)
plt.plot(alist,blist)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.title('Learning for %s' %num_trials)
fig = plt.gcf()
plt.show()
plt.draw()
plt.savefig('ActualVal_Learning_Original.png', dpi=100)
