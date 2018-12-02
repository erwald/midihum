import numpy as np
from keras.layers import LSTM, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam

from data_generator import DataGenerator


def create_model(batch_size):
    print('Setting up model ...')

    # Input shape: 88 notes * 2 states (pressed, sustained) + 14 added features.
    # Output shape: 88 velocities (one for each note).
    number_of_notes = 88
    input_size = number_of_notes * 2 + 14
    output_size = number_of_notes

    # Drop 20% of input units for first layer and 50% for subsequent layers.
    input_dropout = 0.2
    hidden_dropout = 0.5

    model = Sequential()
    model.add(Bidirectional(LSTM(output_size, activation='relu', return_sequences=True, dropout=input_dropout),
                            merge_mode='sum',
                            input_shape=(None, input_size),
                            batch_input_shape=(batch_size, None, input_size)))
    model.add(Bidirectional(LSTM(output_size, activation='relu', return_sequences=True,
                                 dropout=hidden_dropout), merge_mode='sum'))
    model.add(Bidirectional(LSTM(output_size, activation='relu', return_sequences=True,
                                 dropout=hidden_dropout), merge_mode='sum'))
    model.compile(loss='mse', optimizer=Adam(
        lr=1e-3, clipnorm=1), metrics=['mse'])

    print(model.summary())

    return model


def train_model(model, train_names, validate_names, batch_size, epochs,
                model_path, ratify_data, save_model=False, callbacks=[]):
    print('Training model ...')

    train_generator = DataGenerator(
        train_names, batch_size, get_random_augmentation=True, ratify_data=ratify_data)
    validate_generator = DataGenerator(
        validate_names, batch_size, get_random_augmentation=False, ratify_data=ratify_data)

    number_of_train_batches = np.ceil(len(train_names) / float(batch_size))
    number_of_validate_batches = np.ceil(
        len(validate_names) / float(batch_size))

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=number_of_train_batches,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  validation_data=validate_generator,
                                  validation_steps=number_of_validate_batches)

    if save_model:
        print('Saving model ...')
        model.save(model_path)

    return (model, history)


def evaluate(model, test_names, batch_size):
    test_generator = DataGenerator(
        test_names, batch_size, get_random_augmentation=False)
    number_of_test_batches = np.ceil(len(test_names) / float(batch_size))
    return model.evaluate_generator(test_generator, steps=number_of_test_batches)


def predict(model, path, batch_size):
    print('Predicting ...')

    prediction_data = np.load(path)

    # Copy prediction input N times to create a batch of the right size.
    tiled = np.tile(prediction_data, [batch_size, 1, 1])

    raw_prediction = model.predict(tiled, batch_size=batch_size)[0]
    prediction = (raw_prediction * 127).astype(int)  # Float -> MIDI velocity.

    print('Highest predicted velocity: {}'.format(np.max(prediction)))
    print('Lowest predicted velocity: {}'.format(np.min(prediction)))

    return prediction
