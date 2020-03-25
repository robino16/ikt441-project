import config
import logging
import re

# Original text from: https://helsenorge.no/koronavirus/smitte-og-inkubasjonstid
# The text was translated using this online tool: https://www.apertium.org/index.nob.html?dir=nob-nno#translation
# Remember to consider æøå

log = logging.getLogger()


def load_cvs_data():
    # Function to load data file and format the text so it`s ready to be tokenized.
    file = open(config.data_file, 'r', encoding='utf-8')
    for line in file:
        # Get bokmål and nynorsk sentence pair.
        extracted = re.findall(r'<p>.+?</p>', line)

        # Error check. List length should always be 2.
        if len(extracted) != 2:
            log.error('Unexpected number of extracted elements.')  # Should not be possible
            continue

        # Trick to preserve punctuation during training.
        bokmaal = extracted[0].replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace('  ', ' ')
        nynorsk = extracted[1].replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace('  ', ' ')
    return True


def get_clean_sentences(text_in):
    # Function to split inputted text into list of sentences (with some quick cleaning).
    sentences = text_in.lower()
    sentences = sentences.replace('\n', ' ').replace('  ', ' ')  # Remove empty lines
    sentences = sentences.replace('.', '.$').replace(':', ':$').replace('?', '?$').replace('!', '!$')
    return sentences.split('$ ')  # We use $ to split the input into sentences.


def create_csv_data_file():
    # Function to join Bokmål text with Nynorsk translation into a single .csv file.
    log.debug('io_service.py -> create_csv_data_file()')
    log.info('Joining the two files: {} (bokmål) and {} (nynorsk translation) '
             'to a single .csv file: {}.'.format(config.bokmaal_file, config.nynorsk_file, config.data_file))

    bokmaal_file = open(config.bokmaal_file, 'r', encoding='utf-8').read()  # Reads the entire file
    nynorsk_file = open(config.nynorsk_file, 'r', encoding='utf-8').read()

    # Convert to list of sentences.
    bokmaal_sentences = get_clean_sentences(bokmaal_file)
    nynorsk_sentences = get_clean_sentences(nynorsk_file)

    # Error check.
    if len(bokmaal_sentences) != len(nynorsk_sentences):
        log.error('{} contains {} sentences while {} contains {} sentences.'.format(config.bokmaal_file,
                                                                                    len(bokmaal_sentences),
                                                                                    config.nynorsk_file,
                                                                                    len(nynorsk_sentences)))
        log.info('Sometimes the translator forgets to add the last period.')
        return False
    else:
        log.info('Found {} sentences.'.format(len(bokmaal_sentences)))

    # Main loop of this function.
    data_file = open(config.data_file, 'w', encoding='utf-8')
    data_file.write('{}\n'.format(config.data_file_formatting))
    for i in range(len(bokmaal_sentences)):
        data_file.write('{},<p>{}</p>,<p>{}</p>\n'.format(i, bokmaal_sentences[i], nynorsk_sentences[i]))

        # Error check.
        if len(bokmaal_sentences[i].split(' ')) != len(nynorsk_sentences[i].split(' ')):
            log.warning('Sentence #{} has mismatching number of words:'
                        ' {} and {}.'.format(i, len(bokmaal_sentences[i].split(' ')),
                                             len(nynorsk_sentences[i].split(' '))))
            print('Warning: Sentence {} has mismatching number of words.'.format(i))
    data_file.close()
    log.info('.csv data file was created here: {}.'.format(config.data_file))
    print('Info: .csv data file was successfully created.')
    return True


def main():
    print(' --- {} --- '.format(config.io_service_app_name))
    log.info(' --- Running application: {} --- '.format(config.io_service_app_name))

    # Create data file using raw text obtained from the internet.
    if not create_csv_data_file():
        print('Error: Unable to create .csv data file.')
        return False

    # Load the data stored in our data file.
    if not load_cvs_data():
        print('Error: Unable to load .csv data.')
        return False


if __name__ == '__main__':
    main()
