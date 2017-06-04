import argparse
import os
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from utils import INPUT_SHAPE, batch_generator

def load_data(args):
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'), header=None, names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_test, y_train, y_test

def build_model(args):
    # Nvidia self-driving car model

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
    model.add(Dropout(rate=args.keep_prob))
    model.add(Flatten())
    model.add(Dense(units=100, activation='elu'))
    model.add(Dense(units=50, activation='elu'))
    model.add(Dense(units=10, activation='elu'))
    model.add(Dense(units=1))

    return model

def train_model(args, model, X_train, X_test, y_train, y_test):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=args.save_best_only, mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.n_epochs,
#                        workers=4,
#                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_test, y_test, args.batch_size, False),
                        validation_steps=len(X_test),
                        callbacks=[checkpoint],
                        verbose=1)

def s2b(s):
    s = s.lower(s)
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def main():
    parser = argparse.ArgumentParser(description='Self Driving Car Model')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='dropout probability',   dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='n_epochs',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epochs',    dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=bool,  default=True)
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1e-4)

    args = parser.parse_args()

    # print parameters
    print('=' * 30)
    print('Parameters:')
    print('=' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('=' * 30)

    data = load_data(args)
    model = build_model(args)
    model.summary()

    train_model(args, model, *data)

if __name__ == '__main__':
    main()

