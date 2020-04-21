import logging

# Logging module.
log_format = "%(asctime)s - %(levelname)s: %(message)s"
log_output_file = 'output/output.log'
logging.basicConfig(filename=log_output_file, level=logging.DEBUG, format=log_format, filemode='a')
log = logging.getLogger()

# Main application.
main_app_name = 'IKT441 Project - Norwegian Bokmål to Norwegian Nynorsk Translator'

# I/O Service.
io_service_app_name = 'I/O Service for IKT441 Project'
initialize_random_number_generator = False  # For testing purposes.
max_sequence_length = 6
max_nr_of_training_seqs = 50000
max_nr_of_testing_seqs = 1000
augmentation = True

# Data generator.
# Note: Data set filenames can be accessed from data_generator.get_filepath()
url_file = 'data/urls.txt'
min_sentence_length = 15  # A sentence is only valid if it has more characters than this.
aug_seq_len = 4  # All phrases have this length.
training_factor = 0.9
empty_word = '£'

# Model creation.
config_file = 'config.txt'  # Unused.
load_existing_model = False  # Remember: This requires us to use the same tokenizer...
model_save_file = 'output/models/model.{epoch:02d}-{val_loss:.2f}.hdf5'  # File for saving the model.
model_load_file = 'output/models/model.10-3.72.hdf5'  # File for loading the model. Write wanted epoch and loss value.

# Tokenizer.
tokenizer_filter ='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n'

# Model training.
train_model = True
epochs = 30
batch_size = 512
validation_split = 0.2

# Model testing.
output_file = 'output/output.txt'  # Stores original sentences and predicted translated sentences.

# Plotting training history
training_plot_path = 'output/training.png'
