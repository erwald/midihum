# Rachel

Rachel is a recurrent neural network for humanizing MIDI -- that is, for taking as input MIDI compositions with constant velocities (flat loudness/dynamics) and producing as output velocity (or loudness/dynamics) predictions for each of the notes in those compositions.

The theory behind and considerable parts of the code in this project are based on or taken from Iman Malik's thesis project [StyleNet](https://github.com/imalikshake/StyleNet/), about which you may learn more [here](http://imanmalik.com/cs/2017/06/05/neural-style.html).

## How does one use this?

1. Clone this repository.
2. Make sure you have Python 3. Install the required dependencies with `pip` or similar (you can do this later when you try to run the commands below and it complains about whatever it is that you're missing).
3. Download training data (a set can be found towards the end of [the aforementioned blog post](http://imanmalik.com/cs/2017/06/05/neural-style.html)) -- that is, a bunch of MIDI recordings of human performances of compositions in common (4/4) time -- and place them in a `midi_data/` directory.
4. Prepare the training data by running `python main.py --prepare-midi`.
5. Train the model by running `python main.py -t`. After this is done, your model has been saved to `model/model.h5`.
6. You can now load your model with the `-l` flag and either:
    A) predict velocities by running `python main.py -l --predict=/path/to/my/midi.mid`;
    B) or plot a prediction on your training data alongside the true velocities with `python main.py -l --plot=my_midi_file.mid` (said file needs to be one of those in your `midi_data/` directory);
    C) or continue training the model by running `python main.py -lt`.

## How does the model look?

The model is made up of three layers of bidirectional long short-term memory networks (LSTMs), each with a dropout of `0.2`. The first two layers use ReLu as their activation function, the final output layer uses tanh, which for some reason beyond my understanding seems to work best.

It uses Adam (with a learning rate of `0.001` and a gradient clipping `g = 10`) as optimiser. The loss function is mean squared error. That's pretty much it -- for more information, have a look at `model_utility.py`.

## Whence the name?

I named the project after Rachel Heard, whose recording of Haydn's _Andante con variazioni in F minor_ for Naxos remains unsurpassed among the many recordings of that piece I've heard so far, and after Rachel Podger, whose recordings for Channel Classics are a constant joy for me. But I would hardly be surprised to learn that there are many more musical Rachels on our earth whom this project would be glad to claim as its eponyms.
