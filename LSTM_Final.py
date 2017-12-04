import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('Input_Dataset_LSTM.csv', header=0,)
values = dataset.values

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 1
n_features = 6
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 0)


# split into train and test sets
values = reframed.values


n_epoch = 30
w = 0
r = 1
runs = [0]*30

for j in range(0,30):
	n_train_hours = 10000
	train = values[:n_train_hours, :]
	test = values[n_train_hours:, :]
	# split into input and outputs
	n_obs = n_hours * n_features
	train_X, train_y = train[:, :n_obs], train[:,-1]
	test_X, test_y = test[:, :n_obs], test[:, -1]
	# # reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
	test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
	# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
	#
	# # design network
	model = Sequential()
	model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')
	# fit network
	history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
	# plot history
	# pyplot.plot(history.history['loss'], label='train')
	# pyplot.plot(history.history['val_loss'], label='test')
	# pyplot.legend()
	# pyplot.show()

	# # make a prediction
	yhat = model.predict(test_X)


	test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

	# # invert scaling for forecast
	inv_yhat = concatenate((yhat, test_X[:, -6:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]

	# # invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = concatenate((test_y, test_X[:, -6:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]
	count = 0

	for i in range(0,len(inv_yhat)):
		if int(inv_yhat[i]) > 99:
			inv_yhat[i] = 1
		else: inv_yhat[i] = 0
		if int(inv_y[i]) > 99:
			inv_y[i] = 1
		else: inv_y[i] = 0
		if inv_y[i] == inv_yhat[i]:count +=1

	accuracy = (count/len(yhat))*100
	runs[j] = accuracy
	print("Accuracy:"+ str(accuracy))

for i in range(0,len(runs)):
	print("Accuracy:" + str(runs[i]))
mean = sum(runs)/len(runs)
print("n_epoch: {}".format(n_epoch))
print("Mean_Accuracy: {}".format(mean))
print("Standard_Deviation: {}".format(np.std(runs, 0)))

