import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

from model import *

model_output_dir = 'output_model'


def plot_model_history(history, model_name):
    fig = plt.figure(figsize=(14, 11), dpi=180)
    fig.suptitle('Model Performance History', fontsize=10, fontweight='bold')

    plt.plot(history['mean_squared_error'], label='Training MSE')
    plt.plot(history['val_mean_squared_error'], label='Validation MSE')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.legend()

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Write to file.
    output_path = os.path.join(model_output_dir, model_name + ".png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def prepare_input_for_plot(input_data):
    '''Takes an input array and moulds it into something that will look nice
    when plotted. Returning the moulded array.'''

    number_of_note_inputs = 88 * 2

    # Get note data and combine note columns into one for each note.
    input_note_data = input_data[:, :number_of_note_inputs]
    input_note_sustains = [
        (timestep[0::2] + timestep[1::2]) / 2 for timestep in input_note_data]

    # Get other input data.
    input_derived_features_data = input_data[:, number_of_note_inputs:]

    # Combine the columns and transpose so we get timesteps on the x-axis.
    combined_input_data = np.transpose(
        np.hstack((input_note_sustains, input_derived_features_data)))

    return combined_input_data


def plot_prediction(filename, model, batch_size):
    prediction_data_path = os.path.join(
        './input_valid_inputs', filename + '.mid.npy')

    # Load the data and prepare it for plotting.
    input_data = prepare_input_for_plot(np.load(prediction_data_path))
    prediction = np.transpose(predict(
        model, path=prediction_data_path, batch_size=batch_size))

    print('Plotting input and prediction ...')

    fig = plt.figure(figsize=(14, 11), dpi=180)
    fig.suptitle(filename, fontsize=10, fontweight='bold')

    gs = grd.GridSpec(3, 1)

    # Plot input (note on/off).
    ax = fig.add_subplot(gs[0])
    ax.set_title('Input (Note Events and Engineered Features)')
    ax.set_ylabel('Note pitches & other features')
    plt.imshow(input_data, cmap='RdPu', vmin=0, vmax=2,
               origin='lower', interpolation='nearest', aspect='auto')

    # Plot predicted velocities.
    ax = fig.add_subplot(gs[1])
    ax.set_title('Predicted Output (Velocities)')
    ax.set_ylabel('Note pitches')
    plt.imshow(prediction, cmap='jet', vmin=0, vmax=127,
               origin='lower', interpolation='nearest', aspect='auto')

    # Write to file.
    output_path = os.path.join('output', filename.split('.')[0] + ".png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_augmented_sample(filename):
    original_x_path = os.path.join(
        './midi_data_valid_quantized_inputs', f'{filename}.mid.npy')
    x_paths = [original_x_path] + glob.glob(os.path.join(
        './midi_data_valid_quantized_inputs', f'{filename}_*.mid.npy'))
    original_y_path = os.path.join(
        './midi_data_valid_quantized_velocities', f'{filename}.mid.npy')
    y_paths = [original_y_path] + glob.glob(os.path.join(
        './midi_data_valid_quantized_velocities', f'{filename}_*.mid.npy'))

    print('Plotting an augmented sample ...')

    fig = plt.figure(figsize=(32, 24), dpi=300)
    fig.suptitle(filename, fontsize=10, fontweight='bold')

    gs = grd.GridSpec(len(x_paths), 2)

    for idx, (x_path, y_path) in enumerate(zip(x_paths, y_paths)):
        # Load the data and prepare it for plotting.
        input_data = prepare_input_for_plot(np.load(x_path))

        # Plot input (note on/off).
        ax = fig.add_subplot(gs[idx, 0])
        ax.set_title(f'Input {idx + 1}' + (' (Original)' if idx == 0 else ''))
        ax.set_ylabel('Note pitches & other features')
        plt.imshow(input_data, cmap='RdPu', vmin=0, origin='lower',
                   vmax=1, interpolation='nearest', aspect='auto')

        # Plot velocities.
        velocity_data = np.transpose(np.load(y_path))

        ax = fig.add_subplot(gs[idx, 1])
        ax.set_title(f'Expected Output {idx + 1} (Velocities)')
        ax.set_xlabel('Time steps')
        ax.set_ylabel('Note pitches')
        plt.imshow(velocity_data, cmap='jet', vmin=0, vmax=1,
                   origin='lower', interpolation='nearest', aspect='auto')

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Write to file.
    name = filename.split('.')[0]
    output_path = os.path.join(model_output_dir, f'{name}_augmented.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(filename, model, batch_size, suffix=''):
    prediction_data_path = os.path.join(
        './midi_data_valid_quantized_inputs', filename + '.mid.npy')
    true_velocities_path = os.path.join(
        './midi_data_valid_quantized_velocities', filename + '.mid.npy')

    # Load the data and prepare it for plotting.
    input_data = prepare_input_for_plot(np.load(prediction_data_path))

    prediction = np.transpose(predict(
        model, path=prediction_data_path, batch_size=batch_size))
    # Convert raw velocities from float -> MIDI velocity.
    true_velocities = np.transpose(np.load(true_velocities_path)) * 127

    print('Plotting prediction and true velocities ...')

    fig = plt.figure(figsize=(14, 11), dpi=180)
    fig.suptitle(filename, fontsize=10, fontweight='bold')

    gs = grd.GridSpec(3, 1)

    # Plot input (note on/off).
    ax = fig.add_subplot(gs[0])
    ax.set_title('Input (Note Events and Engineered Features)')
    ax.set_ylabel('Note pitches & other features')
    plt.imshow(input_data, cmap='RdPu', vmin=0,
               origin='lower', vmax=1, interpolation='nearest', aspect='auto')

    # Plot predicted velocities.
    ax = fig.add_subplot(gs[1])
    ax.set_title('Predicted Output (Velocities)')
    ax.set_ylabel('Note pitches')
    plt.imshow(prediction, cmap='jet', vmin=0, vmax=127,
               origin='lower', interpolation='nearest', aspect='auto')

    # Plot true velocities.
    ax = fig.add_subplot(gs[2])
    ax.set_title('Expected Output (Velocities)')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Note pitches')
    plt.imshow(true_velocities, cmap='jet', vmin=0, vmax=127,
               origin='lower', interpolation='nearest', aspect='auto')

    # Write to file.
    name = filename.split('.')[0]
    output_path = os.path.join('output', f'{name}{suffix}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
