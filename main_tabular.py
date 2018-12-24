import argparse

import directories
from rachel_tabular import RachelTabular

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prepare-data', action='store_true',
                    help='converts train and validation sets to data frames and saves them')
parser.add_argument('-t', '--train', action='store_true',
                    help='trains the model for the set number of epochs, learning rate and weight decay')
parser.add_argument('--epochs', default=3, type=int, help='epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=0.2, type=float, help='weight decay')
parser.add_argument('--humanize', help='humanizes MIDI file on given path')
args = parser.parse_args()

# Create necessary directories.
directories.create_directories()

rachel = RachelTabular(prepare_data=args.prepare_data)

if args.train:
    rachel.train(args.epochs, args.lr, args.wd)

if args.humanize:
    rachel.humanize(args.humanize)
