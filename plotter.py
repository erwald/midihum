import os
import numpy as np
import matplotlib.pyplot as plt

import model_utility


def plot_comparison(filename, model, batch_size):
    prediction_data_path = os.path.join(
        './midi_data_valid_quantized_inputs', filename + '.npy')
    true_velocities_path = os.path.join(
        './midi_data_valid_quantized_velocities', filename + '.npy')

    prediction = model_utility.predict(
        model, path=prediction_data_path, batch_size=batch_size)
    input_note_data = np.load(prediction_data_path)
    input_note_sustains = [timestep[1::2] for timestep in input_note_data]
    true_velocities = np.load(true_velocities_path)

    print('Plotting prediction and true velocities ...')

    fig = plt.figure(figsize=(14, 11), dpi=120)
    fig.suptitle(filename, fontsize=10, fontweight='bold')

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

    out_png = os.path.join('output', filename.split('.')[0] + ".png")
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
