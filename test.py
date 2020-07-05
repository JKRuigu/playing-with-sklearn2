from sklearn import svm
from sklearn.datasets import load_iris
# data = load_iris() 
from sklearn.model_selection import train_test_split as tts

iris = load_iris()

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2,random_state=4,shuffle=True)

model = svm.SVC(kernel='linear')

# model.fit(X,y)

