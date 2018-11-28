import argparse
import os
import numpy as np
from mido import MidiFile
from keras.models import load_model

# Local imports.
import file_utility
import midi_utility
from model import *
import utility
import plotter
from directories import *
from data_loader import load_data, any_midi_filename
from plot_comparison_callback import PlotComparison

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('--prepare-midi', action='store_true',
                    help='validates, quantizes and saves train and test midi data')
parser.add_argument('--prepare-predictions', action='store_true',
                    help='validates, quantizes and saves prediction midi data')
parser.add_argument('--overwrite-existing', action='store_true',
                    help='prepares midi data even if processed files already exist')
parser.add_argument('--include-baseline', action='store_true',
                    help='also outputs a midi file with static velocities')
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

quantization = 4

args = parser.parse_args()

# Create needed directories if they don't already exist.
create_directories()

if args.prepare_midi:
    print('Preparing train and test MIDI data ...')

    file_utility.validate_data(
        midi_data_path, quantization, args.overwrite_existing)
    file_utility.quantize_data(
        midi_data_valid_path, quantization, args.overwrite_existing)
    file_utility.save_data(midi_data_valid_quantized_path,
                           quantization, args.overwrite_existing)

    plotter.plot_augmented_sample(any_midi_filename())

if args.prepare_predictions:
    print('Preparing prediction MIDI data ...')

    file_utility.validate_data(predictables_dir, quantization)
    file_utility.save_data(predictables_valid_dir, quantization)

# Load the prepared .npy data (validating only if we actually generated any data
# in this session).
#
# Set aside 10% of the data set for validation.
train_names, test_names, x_train, x_test, y_train, y_test = load_data(
    test_size=0.1, random_state=1988, validate=args.prepare_midi)

print('Train sequences: {}'.format(len(x_train)))
print('Test sequences: {}'.format(len(x_test)))

model_name = 'model'
model_path = os.path.join(model_dir, model_name + '.h5')

if args.load_model:
    print('Loading model ...')
    model = load_model(model_path)
else:
    model = create_model(batch_size=args.batch_size)

if args.train_model:
    # Get some MIDI file the prediction for which to plot after each epoch.
    # (It'd be nicer to get one specifically from the validation set, but that's
    # for future me to do.)
    plot_train_comparison_callback = PlotComparison(
        model, any_midi_filename(train_names), args.batch_size, suffix='_train')
    plot_test_comparison_callback = PlotComparison(
        model, any_midi_filename(test_names), args.batch_size, suffix='_test')

    model, history = train_model(model,
                                 x_train=x_train,
                                 y_train=y_train,
                                 x_test=x_test,
                                 y_test=y_test,
                                 batch_size=args.batch_size,
                                 epochs=args.epochs,
                                 model_path=model_path,
                                 save_model=True,
                                 callbacks=[plot_train_comparison_callback, plot_test_comparison_callback])

    # Take metrics and add them to the existing history (iff we loaded the
    # model, iow if we have trained the model before) or use it as a new
    # history, saving the result.
    if args.load_model:
        new_history = np.load('{}.npy'.format(history_dir)).item()
        metrics = ['val_loss', 'val_mean_squared_error',
                   'loss', 'mean_squared_error']
        for metric in metrics:
            new_history[metric] = np.concatenate(
                [new_history[metric], history.history[metric]])
    else:
        new_history = history.history
    np.save(history_dir, new_history)

    # Plot history.
    plotter.plot_model_history(new_history, model_name)

# Evaluate iff we have a model that has at some point been trained.
if args.load_model or args.train_model:
    loss_and_metrics = evaluate(
        model, x_test, y_test, batch_size=args.batch_size)
    print('Final loss and metrics:', loss_and_metrics)


if args.predict:
    prediction_data_path = os.path.join(
        predictables_valid_dir, args.predict + '.npy')
    prediction_midi_path = os.path.join(predictables_dir, args.predict)

    prediction = predict(model, path=prediction_data_path,
                         batch_size=args.batch_size)
    prediction = np.clip(prediction, 0, 1)

    plotter.plot_prediction(args.predict, model=model,
                            batch_size=args.batch_size)

    print('Stylifying MIDI file ...')

    midi_file = MidiFile(prediction_midi_path)
    stylified_midi_file = midi_utility.stylify(
        midi_file, prediction, quantization)

    stylified_midi_file.save(os.path.join(output_dir, args.predict))

if args.include_baseline and args.predict:
    input_data_path = os.path.join(
        predictables_valid_dir, args.predict + '.npy')
    input_midi_path = os.path.join(predictables_dir, args.predict)

    print('Creating baseline (all velocities set to 64) MIDI file ...')

    input_data = np.load(input_data_path)
    input_length = input_data.shape[0]
    output_size = 88  # The number of notes on a grand piano.
    velocities = np.full((input_length, output_size), 0.5)  # Velocity of 63.5.

    midi_file = MidiFile(input_midi_path)
    baseline_midi_file = midi_utility.stylify(
        midi_file, velocities, quantization)

    baseline_midi_file.save(os.path.join(baseline_output_dir, args.predict))

if args.plot:
    plotter.plot_comparison(args.plot, model=model, batch_size=args.batch_size)
