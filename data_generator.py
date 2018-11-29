import numpy as np
import keras
from keras.preprocessing import sequence
from data_loader import load_data


class DataGenerator(keras.utils.Sequence):
    '''Generates data in batches for Rachel.'''

    def __init__(self, names, batch_size, get_random_augmentation, ratify_data=False):
        '''Initialisation.'''
        self.names = names
        self.batch_size = batch_size
        self.get_random_augmentation = get_random_augmentation
        self.ratify_data = ratify_data
        self.on_epoch_end()

    def __len__(self):
        '''Returns the number of batches per epoch.'''
        return int(np.ceil(len(self.names) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data.'''
        # Generate sample start and end indices for the batch index.
        index1 = (index * self.batch_size) % len(self.names)
        index2 = min(index1 + self.batch_size, len(self.names))

        selected_names = self.names[index1:index2]

        # If we have a number of names smaller than our batch size (because we
        # are at the end of the list), complement it with samples from the head
        # of the array.
        sample_shortage = self.batch_size - len(selected_names)
        selected_names = np.concatenate(
            (selected_names, self.names[0:sample_shortage]))

        x, y = load_data(names=selected_names,
                         get_random_augmentation=self.get_random_augmentation,
                         ratify_data=self.ratify_data)

        # Make sure all of the samples in the batch have the same length (by
        # padding the shorter ones with zeroes at the end).
        x = sequence.pad_sequences(x, dtype='float32', padding='post')
        y = sequence.pad_sequences(y, dtype='float32', padding='post')

        return x, y

    def on_epoch_end(self):
        '''Shuffles names after each epoch.'''
        np.random.shuffle(self.names)
