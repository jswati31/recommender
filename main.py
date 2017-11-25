import ml_metrics as metrics
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import os
import load_data as ld
import matplotlib.pyplot as plt
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def autoencoder(x_train, args):
	'''
	Autoencoder
	input shape = feature dimension
	output : autoencoder model
	'''
	encoding_dim = 250
	input_size = x_train.shape[1]
	input_vector = Input(shape=(input_size,))
	encoded = Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.activity_l1(0.01), init='glorot_normal')(input_vector)
	decoded = Dense(input_size, activation='linear', init='glorot_normal')(encoded)
	autoencoder = Model(input_vector, decoded)
	encoder = Model(input_vector, encoded)
	opt = Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon, decay=args.decay)
	autoencoder.compile(optimizer='sgd', loss='mse', metrics=['acc'])
	return autoencoder

def train(model, data, args):
	'''
	Training model
	input : (model, data, arguments)
	output :  trained model
	'''
	(x_train, y_data) = data
	hist = model.fit(x_train, x_train, nb_epoch=args.epochs, batch_size=args.batch_size, validation_split=args.val_split)
	if args.save_weights:
		model.save_weights('model_weights.h5', overwrite=True)
	return hist

def test(model, data, target):
	'''
	Predicts MAP score
	input : (model, data, target)
	output : MAP score
	'''
	(x_test, y_test) = data
	#predicting
	x_decode = model.predict(x_test)
	# Find out the the top 5 hotel cluster predictions
	tmp = x_decode[:, 673:x_decode.shape[1]]
	predictions = [tmp[i].argsort()[-5:][::-1] for i in range(tmp.shape[0])]
	# Calculate the MAP score
	score = metrics.mapk(target, predictions, k=5)

	return score


def plot(hist):
	'''
	plot loss and accuracy curves
	'''
	# Plot the Training and Validation Loss
	plt.plot(hist.history['loss'], label='Training Loss')
	plt.plot(hist.history['val_loss'], label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	legend = plt.legend(loc='upper center', shadow=True)
	plt.show()

	# Plot the Training and Validation Accuracy
	plt.plot(hist.history['acc'], label='Training Accuracy')
	plt.plot(hist.history['val_acc'], label='Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	legend = plt.legend(loc='upper center', shadow=True)
	plt.show()



if __name__ == "__main__":

	# setting the hyper parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
	parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
	parser.add_argument('--val_split', default=0.20, type=float, help='Validation split for validating model')
	parser.add_argument('--is_training', default=1, type=int, help='Training(1) or testing(0)')
	parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate' )
	parser.add_argument('--beta_1', default=0.9, type=float, help='Beta 1')
	parser.add_argument('--beta_2', default=0.999, type=float, help='Beta 2')
	parser.add_argument('--epsilon', default=1e-08, type=float, help='Epsilon')
	parser.add_argument('--decay', default=0.0, type=float, help='Decay rate')
	parser.add_argument('--data_path', default='data/', help='Path to data folder')
	parser.add_argument('--save_weights', default=1, type=int, help='Save weights (Yes=1, No=0)')
	parser.add_argument('--plot', default=1, type=int, help='Plot accuracy or loss curves (Yes=1, No=0)')
	args = parser.parse_args()
	
	#load data
	x_train, x_test, y_data, y_test, target, _ = ld.load()
	
	#define model
	model = autoencoder(x_train, args)
	
	
	# train or test
	if args.is_training:
		hist = train(model=model, data=((x_train, y_data)), args=args)
		if args.plot:
			plot(hist)
	else:  # as long as weights are given, will run testing
		model.load_weights('model_weights.h5')
		score = test(model=model, data=((x_test, y_test)), target=target)
		print('MAP score : ', score)

