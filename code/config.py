import logging

# Logging module.
log_format = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(filename='output.log', level=logging.DEBUG, format=log_format, filemode='a')

# Main application.
main_app_name = 'IKT441 Project - Norwegian Bokmål to Norwegian Nynorsk Translator'

# I/O Service.
io_service_app_name = 'I/O Service for IKT441 Project (testing)'
bokmaal_file = 'data/bokmaal.txt'
nynorsk_file = 'data/nynorsk.txt'
data_file = 'data/data.txt'  # Dataset in csv format.
data_file_formatting = 'id,<p>norwegian_bokmaal</p>,<p>norwegian_nynorsk</p>'
