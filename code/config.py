import logging

# Logging module.
log_format = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(filename='output.log', level=logging.DEBUG, format=log_format, filemode='a')
log = logging.getLogger()

# Main application.
main_app_name = 'IKT441 Project - Norwegian Bokmål to Norwegian Nynorsk Translator'

# I/O Service.
io_service_app_name = 'I/O Service for IKT441 Project (testing)'
text_file_original = 'data/bokmaal.txt'
text_file_translated = 'data/nynorsk.txt'
data_file = 'data/data.txt'  # Dataset in csv format.
data_temp_file = 'data/temp.txt'  # Used when producing/validating new dataset.
data_file_formatting = 'id,<p>norwegian_bokmaal</p>,<p>norwegian_nynorsk</p>'
training_factor = 0.9
initialize_random_number_generator = False  # For testing purposes.

# Tokenizer.
tokenizer_file_original = 'tokenizer_original.pickle'
tokenizer_file_translated = 'tokenizer_translated.pickle'
load_tokenizers = False  # Currently not supported as there is no word count check going on.
tokenizer_filter ='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n'

# Model creation.
config_file = 'config.txt'  # Unused.
load_existing_weights = True  # Remember: This requires us to use the same tokenizer...
weights_file = 'model_weights.h5'  # File for storing model weights.

# Model training.
save_weights = True
train_weights = True
epochs = 30
batch_size = 32
validation_split = 0.1

# Model testing.
output_file = 'output/output.txt'  # Stores original sentences and predicted translated sentences.

# Plotting training history
training_plot_path = 'output/training.png'
