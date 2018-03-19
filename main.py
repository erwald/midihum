import argparse
import os
import numpy as np
from mido import MidiFile
from keras.models import load_model
from sklearn.cross_validation import train_test_split

import file_utility
import midi_utility
import model_utility
import utility
import plotter

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('--prepare-midi', action='store_true',
                    help='validates, quantizes and saves train and test midi data')
parser.add_argument('--prepare-predictions', action='store_true',
                    help='validates, quantizes and saves prediction midi data')
parser.add_argument('--batch-size', default=4, type=int, help='batch size')
parser.add_argument('--epochs', default=1, type=int, help='batch size')
parser.add_argument(
    '--predict', help='make prediction for midi file on given path')
parser.add_argument(
    '--plot', help='make prediction and plot it compared with known answer')
parser.add_argument('-l', '--load-model', action='store_true',
                    help='loads a model from disk')
parser.add_argument('-t', '--train-model', action='store_true',
                    help='trains the model for the set number of epochs')


def load_data():
    '''Loads the musical performances and returns sets of inputs and labels
    (notes and resulting velocities), one for testing and one for training.'''

    print('Loading data ...')

    midi_data_inputs_path = './midi_data_valid_quantized_inputs'
    midi_data_velocities_path = './midi_data_valid_quantized_velocities'

    def maybe_add_name(filename):
        return filename[1] if filename[1].split('.')[-1] == 'npy' else None

    names = utility.map_removing_none(
        maybe_add_name, enumerate(os.listdir(midi_data_inputs_path)))

    # N songs of Mn timesteps, each with 176 (= 88 * 2) pitch classes.
    # Iow, each data point: [Mn, 176]
    input_data = []

    # N songs of Mn timesteps, each with 88 velocities.
    # Iow, each data point: [Mn, 88]
    velocity_data = []

    for i, filename in enumerate(names):
        abs_inputs_path = os.path.join(midi_data_inputs_path, filename)
        abs_velocities_path = os.path.join(midi_data_velocities_path, filename)
        loaded_inputs = np.load(abs_inputs_path)

        input_data.append(loaded_inputs)

        loaded_velocities = np.load(abs_velocities_path)
        loaded_velocities = loaded_velocities / 127  # MIDI velocity -> float.

        velocity_data.append(loaded_velocities)

        assert input_data[i].shape[0] == velocity_data[i].shape[0]

    return train_test_split(input_data, velocity_data, test_size=0.05)


args = parser.parse_args()

midi_data_path = './midi_data'
midi_data_valid_path = './midi_data_valid'
midi_data_valid_quantized_path = './midi_data_valid_quantized'

predictables_path = './input'
predictables_valid_path = './input_valid'

data_path = './data'
training_set_path = os.path.join(data_path, 'training')
velocities_path = os.path.join(data_path, 'velocities')
validation_set_path = os.path.join(data_path, 'validation')

quantization = 4

if args.prepare_midi:
    print('Preparing train and test MIDI data ...')

    file_utility.validate_data(midi_data_path, quantization)
    file_utility.quantize_data(midi_data_valid_path, quantization)
    file_utility.save_data(midi_data_valid_quantized_path, quantization)

if args.prepare_predictions:
    print('Preparing prediction MIDI data ...')

    file_utility.validate_data(predictables_path, quantization)
    file_utility.save_data(predictables_valid_path, quantization)

x_train, x_test, y_train, y_test = load_data()

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


model_name = 'model'
model_path = 'models/' + model_name + '.h5'

if args.load_model:
    print('Loading model ...')
    model = load_model(model_path)
else:
    model = model_utility.create_model(batch_size=args.batch_size)

if args.train_model:
    model, history = model_utility.train_model(model,
                                               x_train=x_train,
                                               y_train=y_train,
                                               batch_size=args.batch_size,
                                               epochs=args.epochs,
                                               model_path=model_path,
                                               save_model=True)
    plotter.plot_model_history(history, model_name)

# Evaluate.
loss_and_metrics = model_utility.evaluate(
    model, x_test, y_test, batch_size=args.batch_size)
print('Loss and metrics:', loss_and_metrics)


if args.predict:
    prediction_data_path = os.path.join(
        './input_valid_inputs', args.predict + '.npy')
    prediction_midi_path = os.path.join('./input', args.predict)

    prediction = model_utility.predict(
        model, path=prediction_data_path, batch_size=args.batch_size)
    prediction = np.clip(prediction / 127.0, 0, 1)

    print('Stylifying MIDI file ...')

    midi_file = MidiFile(prediction_midi_path)
    stylified_midi_file = midi_utility.stylify(
        midi_file, prediction, quantization)

    stylified_midi_file.save(os.path.join('./output', args.predict))

if args.plot:
    plotter.plot_comparison(args.plot, model=model, batch_size=args.batch_size)
