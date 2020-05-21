import logging

# Logging module.
log_format = "%(asctime)s - %(levelname)s: %(message)s"
log_output_file = 'output/output.log'
logging.basicConfig(filename=log_output_file, level=logging.DEBUG, format=log_format, filemode='a')
log = logging.getLogger()

# Main application.
main_app_name = 'IKT441 Project - Norwegian Bokmål to Norwegian Nynorsk Translator'
test_app_name = 'IKT441 Project - Testing'
use_full_sentences = False

# I/O Service.
io_service_app_name = 'I/O Service for IKT441 Project'
initialize_random_number_generator = False  # For testing purposes.
max_nr_of_training_seqs = 300000  # 300000 is a good value
max_nr_of_testing_seqs = 100
augmentation = True
full_sequence_length = 15
input_sequence_length = full_sequence_length if use_full_sentences else 4
output_sequence_length = full_sequence_length if use_full_sentences else 6

# Data generator.
# Note: Data set filenames can be accessed from data_generator.get_filepath()
url_file = 'data/urls.txt'
min_sentence_length = 15  # A sentence is only valid if it has more characters than this.
aug_seq_len = 15  # All phrases have this length.
training_factor = 0.9
empty_word = '£'

# Model creation.
config_file = 'config.txt'  # Unused.
load_existing_model = False  # Remember: This requires us to use the same tokenizer...
model_folder = 'models_full' if use_full_sentences else 'models'
model_save_file = 'output/' + model_folder + '/model.{epoch:02d}-{val_loss:.2f}.hdf5'
model_load_file = 'output/models/model.30-1.75.hdf5'  # File for loading the model. Write wanted epoch and loss value.

# Tokenizer.
tokenizer_filter ='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n'

# Model training.
train_model = True
epochs = 14 if use_full_sentences else 30
batch_size = 34 if use_full_sentences else 512
validation_split = 0.1

# Model testing.
output_file = 'output/full/output.txt'if use_full_sentences else 'output/output.txt'

# Plotting training history
training_plot_path = 'output/full/training.png' if use_full_sentences else 'output/training.png'


def get_conf():
    conf = 'Configuration:\n' \
           'Training epochs: {} epochs.\n' \
           'Input sequence length: {}.\n' \
           'Output sequence length: {}.\n' \
           'Training sequences: {}.\n' \
           'Batch size: {}.\n' \
           'Validation split: {}.\n'.format(epochs,
                                            training_factor,
                                            input_sequence_length,
                                            output_sequence_length,
                                            max_nr_of_training_seqs,
                                            batch_size,
                                            validation_split)

    return conf
