import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1 : Data Preparation
df = pd.read_csv('train.csv')
# print(df.shape)

data = df.values
X = data[:, 1:]
Y = data[:, 0]

split = int(0.8*X.shape[0])
print(split)

X_train = X[:split, :]
Y_train = Y[:split]
X_test = X[split:, :]
Y_test = Y[split:]
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Visualising some samples
def drawImg(sample):
    img = sample.reshape((28, 28))
    plt.imshow(img, cmap = 'gray')
    plt.show()

drawImg(X_train[3])
print(Y_train[3])

# Step 2 : K-NN
def dist(X1, X2):
    return np.sqrt(sum((X1 - X2)**2))

def knn(X, Y, queryPoint, k = 5):
    vals = []
    m = X.shape[0]

    for i in range(m):
        d = dist(queryPoint, X[i])
        vals.append((d, Y[i]))

    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    vals = np.array(vals)
    # print(vals)

    new_vals = np.unique(vals[:, 1], return_counts = True)
    print(new_vals)

    index = new_vals[1].argmax()
    pred = new_vals[0][index]

    return pred

# Step 3 : Make Predictions
pred = knn(X_train, Y_train, X_test[0])
print(int(pred))

drawImg(X_test[0])
print(Y_test[0])