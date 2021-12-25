import pandas as pd
import numpy as np
df = pd.read_csv('potato.csv')

labels=df['p/kg']
relative_change=[]
temp=[]
for i in range(1,len(labels)):
    result=(labels[i]-labels[i-1])/labels[i-1]
    relative_change.append(result)
    label=-2
    if result>0:
        label=1
    elif result==0:
        label=0
    else:
        label=-1
    temp.append(label)
relative_change.append(0)
new_temp = temp+[0]
df['yms']=new_temp
df['relative_change']=relative_change

from sklearn.model_selection import train_test_split
training_data,testing_data=train_test_split(df,test_size=0.2,shuffle=False)
print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")

X_train,Y_train,X_test,Y_test=training_data['p/kg'],training_data['yms'],testing_data['p/kg'],testing_data['yms']


def euclidean_dist(X_test, X_train):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    # a = np.zeros(num_test)
    # b = np.zeros(num_train)
    # for i in range(num_test):
    #     a[i] = np.dot(X_test[i], X_test[i])
    # for j in range(num_train):
    #     b[j] = np.dot(X_train[j], X_train[j])  
    for i in range(num_test):
        for j in range(num_train):
            # print('X_test[i]=',X_test.iloc[i],'X_train[j]=',X_train.iloc[j])
            dists[i, j] = np.sqrt(np.square(X_test.iloc[i])+np.square(X_train.iloc[j]))


    return dists
    # dists = np.add(np.sum(X_test ** 2, axis=1, keepdims=True), np.sum(X_train ** 2, axis=1, keepdims=True).T) - 2* X_test @ X_train.T
    # return dists


# [[ 8 10  1]
#  [ 2  8  9]]
# x_train = np.array([[1, 2], [0, 3], [-1, 1]])
# x_test = np.array([[-1, 0], [2, 1]])
# print(X_test.iloc[0])
# print(X_train.shape)

my_dists = euclidean_dist(X_test, X_train)
print(my_dists.shape)

# sorted_idx = np.argsort(my_dists,axis=1)
# print(sorted_idx[0])
# print('---------')
# print(sorted_idx[20])
# # k neareast neighbour
# def find_k_neighbors(dists, Y_train, k):
#     num_test = dists.shape[0]
#     neighbors = np.zeros((num_test, k))
#     sorted_idx = np.argsort(dists,axis=1)
#     for i in range(num_test):
#         neighbors[i] = Y_train.loc[sorted_idx[i][:k]]
#     return sorted_idx

# neighbors=find_k_neighbors(my_dists,Y_train,22)
# print(neighbors)


# def knn_predict(X_test, X_train, Y_train, k):
#     num_test = X_test.shape[0]
#     Y_pred = np.zeros(num_test, dtype=int)
#     dists = euclidean_dist(X_test, X_train)
#     neighbors = find_k_neighbors(dists, Y_train, k)

#     for i in range(num_test):
#         value, counts = np.unique(neighbors[i], return_counts=True)
#         idx = np.argmax(counts)
#         Y_pred[i] = value[idx]
    
#     return Y_pred


# k=5
# Y_pred = knn_predict(X_test, X_train, Y_train, k)
# print(Y_pred)