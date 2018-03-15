import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

import model_utility


def plot_comparison(filename, model, batch_size):
    prediction_data_path = os.path.join(
        './midi_data_valid_quantized_inputs', filename + '.npy')
    true_velocities_path = os.path.join(
        './midi_data_valid_quantized_velocities', filename + '.npy')

    # Load the data and transpose so we get timesteps on the x-axis.
    input_note_data = np.load(prediction_data_path)
    input_note_sustains = np.transpose(
        [timestep[1::2] for timestep in input_note_data])
    prediction = np.transpose(model_utility.predict(
        model, path=prediction_data_path, batch_size=batch_size))
    true_velocities = np.transpose(np.load(true_velocities_path))

    print('Plotting prediction and true velocities ...')

    fig = plt.figure(figsize=(14, 11), dpi=180)
    fig.suptitle(filename, fontsize=10, fontweight='bold')

    gs = grd.GridSpec(3, 1)

    # Plot input (note on/off).
    fig.add_subplot(gs[0])
    plt.imshow(input_note_sustains, cmap='binary', vmin=0,
               vmax=1, interpolation='nearest', aspect='auto')

    # Plot predicted velocities.
    fig.add_subplot(gs[1])
    plt.imshow(prediction, cmap='jet', vmin=0, vmax=127,
               interpolation='nearest', aspect='auto')

    # Plot true velocities.
    fig.add_subplot(gs[2])
    plt.imshow(true_velocities, cmap='jet', vmin=0, vmax=127,
               interpolation='nearest', aspect='auto')

    out_png = os.path.join('output', filename.split('.')[0] + ".png")
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
