# Importing dependencies.
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


# Creation of the main perceptron object.
class Perceptron(object):
    # Initiating the learning rate and number of iterations.
    # Q1. 学習率とIterationの回数はどのように設定するのが適切なのか？
    #　Q2. 最後のself.weights = np.zeros(1 + x.shape[1]) 「重みづけ」を定義している箇所と理解しているが、
    #　.zerosで配列を初期化しているところまでは理解できたが、（）の中の式の意味は？
    # .shape:配列の大きさを取得。.shape[1]なので、“x配列”の列の大きさを取得している。
    def __init__(self, Learn_Rate=0.5, Iterations=10):
        self.learn_rate = Learn_Rate
        self.Iterations = Iterations
        self.errors = []
        self.weights = np.zeros(1 + x.shape[1])

    # Defining fit method for model training.（要するに学習の仕方を定義している）
    #　Q3. classの中でも"self.weights = np.zeros(1 + x.shape[1])"を定義しているが、再度ここで定義している意味は？
    #　zip関数を使用して、ｘ・ｙを同時に取得
    def fit(self, x, y):
        self.weights = np.zeros(1 + x.shape[1])
        for i in range(self.Iterations):
            error = 0
            for xi, target in zip(x, y):
                update = self.learn_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi #weightsの開始位置：１から順に取得
                self.weights[0] += update #weightsの開始位置0から順に取得
                error += int(update != 0) #updateが０と等しいかどうか確認
            self.errors.append(error)
        return self
    # Q4. ここは何を定義しているのか？　=> 行列の内積計算
    def net_input(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    # predict（予測）メソッドを定義
    def predict(self,x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

# Data retrieval and preparation.
# iloc: iloc は 行、列を番号で指定します（先頭が 0）
# Dataは、150行(instance)、4列(attribute)、ヘッダー無し。
# データは、1～50がIris-Setosa,51～100がIris-versicolor,101～150がIris-virginica
y = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None)
x = y.iloc[0:100, [0, 2]].values
plt.scatter(x[:50, 0], x[:50, 1], color='red')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue')
plt.scatter(x[100:150, 0], x[100:150, 1], color='yellow')
# plt.scatter: 散布図を描く方法！
plt.show()
y = y.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Model training and evaluation.
Classifier = Perceptron(Learn_Rate=0.01, Iterations=50)
Classifier.fit(x, y)
plt.plot(range(1, len(Classifier.errors) + 1), Classifier.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()




# Defining function that plots the decision regions.
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


# Showing the final results of the perceptron model.
plot_decision_regions(x, y, classifier=Classifier)
plt.show()