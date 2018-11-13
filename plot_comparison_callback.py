from keras.callbacks import Callback
import datetime
import plotter


class PlotComparison(Callback):
    def __init__(self, model, sample, batch_size):
        super(PlotComparison, self).__init__()
        self.model = model
        self.sample = sample
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        date_string = '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        plotter.plot_comparison(self.sample,
                                model=self.model,
                                batch_size=self.batch_size,
                                suffix=date_string)
