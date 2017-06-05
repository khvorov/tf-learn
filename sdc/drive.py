import argparse
import base64
import eventlet
import eventlet.wsgi
import h5py
import numpy as np
import os
import shutil
import socketio
import utils

from PIL import Image
from flask import Flask
from io import BytesIO
from keras import __version__ as keras_version
from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None

MIN_SPEED, MAX_SPEED = 2, 15
speed_limit = MAX_SPEED

@sio.on('connect')
def connect(sid, env):
    print('connected ', sid)
    send_control(0, 0)

@sio.on('telemetry')
def telemetry(sid, data):
    if not data:
        sio.emit('manual', data={}, sid=True)
        return

    steering_angle = data['steering_angle']
    throttle = data['throttle']
    speed = float(data['speed'])

    # get the image
    image = np.asarray(Image.open(BytesIO(base64.b64decode(data['image']))))
    image = utils.preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image, batch_size=1))

    global speed_limit
    speed_limit = MIN_SPEED if speed > speed_limit else MAX_SPEED
    throttle = 1.0 - steering_angle**2 - (speed / speed_limit)**2

    print('{} {} {}'.format(steering_angle, throttle, speed))

    send_control(steering_angle, throttle)

def send_control(steering_angle, throttle):
    sio.emit('steer',
            data={
                'steering_angle': steering_angle.__str__(),
                'throttle': throttle.__str__()
            },
            skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, help='Path to model h5 file')
    parser.add_argument('image_folder', type=str, nargs='?', default='', help='Path to image folder. This is where the images will be saved')
    args = parser.parse_args()

    # load model
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('Version mismatch - keras version: ', keras_version, ', model version: ', model_version)

    model = load_model(args.model)

    # save images
    if args.image_folder != '':
        print('Creating image folder at {}'.format(args.image_folder))
        if os.path.exists(args.image_folder):
            shutil.rmtree(args.image_folder)
        os.makedirs(args.image_folder)
        print('RECORDING THIS RUN...')
    else:
        print('NOT RECORDING THIS RUN...')

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

