import os

# Data
midi_data_path = 'midi_data'
midi_data_valid_path = 'midi_data_valid'
midi_data_valid_quantized_path = 'midi_data_valid_quantized'

# Model.
model_dir = 'models'
history_dir = os.path.join(model_dir, 'history')

# Input (for predictions).
predictables_dir = 'input'
predictables_valid_dir = 'input_valid'

# Artefacts.
output_dir = 'output'
model_output_dir = 'output_model'
baseline_output_dir = 'output_baseline'


def create_directories():
    dirs = [model_dir, history_dir, output_dir,
            model_output_dir, baseline_output_dir]
    [os.makedirs(d) for d in dirs if not os.path.exists(d)]
