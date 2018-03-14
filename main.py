from __future__ import print_function
import file_utility
import midi_utility
import utility
import model
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mido import MidiFile
from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import LSTM, Bidirectional
from keras.optimizers import Adam

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
parser.add_argument('-s', '--save-model', action='store_true',
                    help='saves the model to file')
parser.add_argument('-l', '--load-model', action='store_true',
                    help='loads a model from disk')

model_path = 'model.h5'


def load_data():
    '''Loads the musical performances and returns sets of inputs and labels
    (notes and resulting velocities), one for testing and one for training.'''

    print('Loading numpy data ...')

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
    file_utility.save_data(midi_data_valid_quantized_path,
                           quantization, one_hot=True)

if args.prepare_predictions:
    print('Preparing prediction MIDI data ...')

    file_utility.validate_data(predictables_path, quantization)
    file_utility.save_data(predictables_valid_path, quantization, one_hot=True)

x_train, x_test, y_train, y_test = load_data()

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

number_of_notes = 88
input_size = number_of_notes * 2  # 88 notes * 2 states (pressed, sustained).
output_size = number_of_notes  # 88 notes.

max_sample_length = len(max(x_train, key=len))


def batch_generator(xs, ys, batch_size=args.batch_size):
    '''Generates a batch of samples for training or validation.'''
    i = 0
    while True:
        index1 = (i * batch_size) % len(xs)
        index2 = min(index1 + batch_size, len(xs))
        x, y = xs[index1:index2], ys[index1:index2]
        x = sequence.pad_sequences(x, dtype='float32', padding='post')
        y = sequence.pad_sequences(y, dtype='float32', padding='post')
        yield (x, y)


if args.load_model:
    print('Loading model ...')

    model = load_model(model_path)

else:
    print('Setting up model ...')

    dropout = 0.2  # Drop 20% of units for linear transformation of inputs.

    model = Sequential()
    model.add(Bidirectional(LSTM(output_size, activation='relu', return_sequences=True, dropout=dropout),
                            merge_mode='sum',
                            input_shape=(None, input_size),
                            batch_input_shape=(args.batch_size, None, input_size)))
    model.add(Bidirectional(LSTM(output_size, activation='relu',
                                 return_sequences=True, dropout=dropout), merge_mode='sum'))
    model.add(Bidirectional(LSTM(output_size, activation='tanh',
                                 return_sequences=True, dropout=dropout), merge_mode='sum'))
    model.compile(loss='mse', optimizer=Adam(
        lr=0.001, clipnorm=10), metrics=['mse'])

    print(model.summary())

    print('Training model ...')

    number_of_train_batches = np.ceil(len(x_train)/float(args.batch_size))
    model.fit_generator(batch_generator(x_train, y_train),
                        steps_per_epoch=number_of_train_batches,
                        epochs=args.epochs)

    padded_x_test = sequence.pad_sequences(
        x_test, dtype='float32', padding='post')
    padded_y_test = sequence.pad_sequences(
        y_test, dtype='float32', padding='post')

    if args.save_model:
        print('Saving model ...')
        model.save(model_path)

    number_of_test_batches = np.ceil(len(padded_x_test)/float(args.batch_size))
    loss_and_metrics = model.evaluate_generator(batch_generator(padded_x_test, padded_y_test),
                                                steps=number_of_test_batches)
    print('Loss and metrics:', loss_and_metrics)


def predict(path):
    print('Predicting ...')

    prediction_data = np.load(path)
    np.savetxt('output/prediction_data.out', prediction_data, fmt='%d')

    # Copy prediction input N times to create a batch of the right size.
    tiled = np.tile(prediction_data, [args.batch_size, 1, 1])

    raw_prediction = model.predict(tiled, batch_size=args.batch_size)[0]
    prediction = (raw_prediction * 127).astype(int)  # Float -> MIDI velocity.

    print('Max:', np.max(prediction))
    print('Min:', np.min(prediction))
    np.savetxt('output/prediction_result.out', prediction, fmt='%d')

    return prediction


if args.predict:
    prediction_data_path = os.path.join(
        './input_valid_inputs', args.predict + '.npy')
    prediction_midi_path = os.path.join('./input', args.predict)

    prediction = predict(prediction_data_path)
    # TODO: Figure out way of using full range.
    prediction = np.clip(prediction / 127.0, 0, 1)

    print('Stylifying MIDI file ...')

    midi_file = MidiFile(prediction_midi_path)
    stylified_midi_file = midi_utility.stylify(
        midi_file, prediction, quantization)

    stylified_midi_file.save(os.path.join('./output', args.predict))

if args.plot:
    prediction_data_path = os.path.join(
        './midi_data_valid_quantized_inputs', args.plot + '.npy')
    true_velocities_path = os.path.join(
        './midi_data_valid_quantized_velocities', args.plot + '.npy')

    prediction = predict(prediction_data_path)
    input_note_data = np.load(prediction_data_path)
    input_note_sustains = [timestep[1::2] for timestep in input_note_data]
    true_velocities = np.load(true_velocities_path)

    print('Plotting prediction and true velocities ...')

    fig = plt.figure(figsize=(14, 11), dpi=120)
    fig.suptitle(args.plot, fontsize=10, fontweight='bold')

    # Plot input (note on/off).
    fig.add_subplot(1, 3, 1)
    plt.imshow(input_note_sustains, cmap='binary', vmin=0,
               vmax=1, interpolation='nearest', aspect='auto')

    # Plot predicted velocities.
    fig.add_subplot(1, 3, 2)
    plt.imshow(prediction, cmap='jet', vmin=0, vmax=127,
               interpolation='nearest', aspect='auto')

    # Plot true velocities.
    fig.add_subplot(1, 3, 3)
    plt.imshow(true_velocities, cmap='jet', vmin=0, vmax=127,
               interpolation='nearest', aspect='auto')

    out_png = os.path.join('output', args.plot.split('.')[0] + ".png")
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
