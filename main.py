import matplotlib.pyplot as plt 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(hidden_layer_sizes=(25) , activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1, verbose=True)


#hidden_layer_sizes: 25; chosen for quick testing purposes
#activation: logistic because the problem involves values between 0 and 1
#alpha: learning rate; chosen arbitrarily
#solver: optimizer; chose basic gradient descent
#tol: tolerance; when loss doesn't change by 1e-4 or more, it stops
#learning_rate_init: initial learning rate
#verbose: prints progress messages

digit = datasets.load_digits()
x = digit.images.reshape((len(digit.images), -1)) #flattens pixels
y = digit.target


#training data
trainx = x[:1000]
trainy = y[:1000]

#testing data
testx = x[1000:]
testy = y[1000:]

mlp.fit(trainx,trainy)

prediction = mlp.predict(testx)
prediction[:50] 

testy[:50] 

accuracy_score(testy, prediction)