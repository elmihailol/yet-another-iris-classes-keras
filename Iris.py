# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# https://matplotlib.org/users/pyplot_tutorial.html
import matplotlib.pyplot as plt
import numpy
import pandas
import operator
from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

def find_max(x):
    max_index, max_value = max(enumerate(x), key=operator.itemgetter(1))
    return max_index
    # return max(x)

numpy.random.seed(0)

# Читаем ирисы через пандас
dataframe = pandas.read_csv("iris.csv", header=None)
# Конвертируем dataframe в "контейнеры", связаны
data = dataframe.values
print(type(data))
numpy.random.shuffle(data)
print(data)
# Достаем X и Y, связаны
X = data[:,0:4].astype(float)
y = data[:,4]
print(type(X))
# Создаем энкодер для названий, связаны
encoder = LabelEncoder()
# Классификация, связаны
encoder.fit(y)
# Трансформируемся, связаны
encoded_Y = encoder.transform(y)
# У нас есть вектор числе типа [0,0,0,.....,1,1,1,.....,2,2,2,2] нужно его преобразовать [[ 1.  0.  0.],....,
# [ 0.  1.  0.],....., [ 0.  0.  1.]], связаны
dummy_y = np_utils.to_categorical(encoded_Y)

# Создамка тренеровочные данные и тестовые, чтоб все как у людей
persent = 0.8
len_train = int(len(X)*persent)
trainX = X[0:len_train]
trainY = dummy_y[0:len_train]
testX =  X[0:-len_train]
testY = dummy_y[0:-len_train]

# Создаем модель сети, для кого я это пишу вообще?
model = Sequential()
model.add(Dense(6, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=1, epochs=10,verbose=1)
# Делаем предсказания класса на основе testX
prediction = model.predict(testX)
# Вычисляем ошибку
eval = model.evaluate(testX,testY,verbose=1)
print(len(trainX))
print(len(testX))
print(eval)
# Подготовливаем данные для графика
graph_prediction = []
for i in range(len(prediction)):
    graph_prediction.append(find_max(prediction[i]))
# Просто очень нравится list'ы
graph_real = []
for i in range(len(testY)):
    graph_real.append(find_max(testY[i]))

plt.plot(graph_prediction)
plt.plot(graph_real)
plt.ylabel('Class')
plt.xlabel('Number of flower')
plt.title('Loss value = ' + str(eval[0]))
plt.show()