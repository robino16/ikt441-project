import config
import logging
import re
import sentence
from keras.preprocessing.text import Tokenizer
import pickle

# Original text from: https://helsenorge.no/koronavirus/smitte-og-inkubasjonstid
# The text was translated using this online tool: https://www.apertium.org/index.nob.html?dir=nob-nno#translation
# Remember to consider æøå

log = logging.getLogger()

# todo: Tokenizing all words and checking if existing tokenizer object can be reused.
# todo: Function that splits training and testing data


def save_tokenizer(tokenizer_in, filename_in):
    # This function is responsible for saving the tokenizer to a pickle-object so it can be reused.
    with open(filename_in, 'wb') as handle:
        pickle.dump(tokenizer_in, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(filename_in):
    # Function responsible of loading existing tokenizer file.
    try:
        with open(filename_in, 'rb') as handle:
            tokenizer = pickle.load(handle)
            log.debug('Successfully loaded tokenizer object from file: {}.'.format(filename_in))
            return tokenizer
    except:
        log.warning('Failed to load tokenizer object from file: {}.'.format(filename_in))
        print('Warning: Failed to load tokenizer object from file {}.'.format(filename_in))
        return None


def get_tokenizer(filename_in, sentences_in):
    # Returns a single tokenizer object.
    tokenizer = None
    if config.load_tokenizers:
        # Load existing tokenizer from file.
        tokenizer = load_tokenizer(filename_in)
    if tokenizer is None:
        # Create new tokenizer.
        tokenizer = Tokenizer(filters=config.tokenizer_filter)
        tokenizer.fit_on_texts(sentences_in)
        save_tokenizer(tokenizer, filename_in)  # Save it on disk.
    total_words = len(tokenizer.word_index) + 1
    log.debug('Tokenizer ({}) has {} indexed words.'.format(filename_in, total_words))
    return tokenizer


def get_tokenizers(original_sentences_in, translated_sentences_in):
    # This function returns the tokenizer objects.
    # Note: There are two tokenizer objects, one for the original language and one for the translation.
    tokenizer_original = get_tokenizer(config.tokenizer_file_original, original_sentences_in)
    tokenizer_translated = get_tokenizer(config.tokenizer_file_translated, translated_sentences_in)
    return tokenizer_original, tokenizer_translated


def get_sentence_objects(original_sentences_in, translated_sentences_in):
    # Returns a list of Sentence objects (containing both original and translated version of a sentence).
    # Note: This functionality is currently ignored. 

    # Error check
    if len(original_sentences_in) != len(translated_sentences_in):
        log.error('Number of sentences are not equal: {} (original) '
                  'and {} (translated).'.format(len(original_sentences_in), len(translated_sentences_in)))
        return None

    sentences = []
    for i in range(original_sentences_in):
        sentences.append(sentence.Sentence(i, original_sentences_in, translated_sentences_in))
    return sentences


def load_cvs_data():
    # Function to load data file and format the text so it`s ready to be tokenized.
    original_sentences = []
    translated_sentences = []

    file = open(config.data_file, 'r', encoding='utf-8')
    for line in file:
        # Get original and translated version of the sentence.
        extracted = re.findall(r'<p>.+?</p>', line)

        # Error check. List length should always be 2 (one original and one translated).
        if len(extracted) != 2:
            log.error('Unexpected number of extracted elements.')  # Should not be possible
            continue

        # Trick to preserve punctuation during training.
        original = extracted[0].replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace('  ', ' ')
        translated = extracted[1].replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace('  ', ' ')

        original_sentences.append(original)
        translated_sentences.append(translated)
    file.close()

    return original_sentences, translated_sentences


def split_text_to_sentences(text_in):
    # Function to split inputted text into list of sentences (with some quick cleaning).
    sentences = text_in.lower()
    sentences = sentences.replace('\n', ' ').replace('  ', ' ')  # Remove empty lines

    # I preserve punctuation.
    sentences = sentences.replace('.', '.$').replace(':', ':$').replace('?', '?$').replace('!', '!$')
    return sentences.split('$ ')  # We use $ to split the input into sentences.


def create_csv_data_file():
    # Function to join original text with translation into a single .csv file.
    log.debug('io_service.py -> create_csv_data_file()')
    log.info('Joining the two files: {} (original) and {} (translated) '
             'to a single .csv file: {}.'.format(config.text_file_original, config.text_file_translated, config.data_file))

    original_text = open(config.text_file_original, 'r', encoding='utf-8').read()  # Reads the entire file
    translated_text = open(config.text_file_translated, 'r', encoding='utf-8').read()

    # Convert to list of sentences.
    original_sentences = split_text_to_sentences(original_text)
    translated_sentences = split_text_to_sentences(translated_text)

    # Error check.
    if len(original_sentences) != len(translated_sentences):
        log.error('{} contains {} sentences while {} contains {} sentences.'.format(config.text_file_original,
                                                                                    len(original_sentences),
                                                                                    config.text_file_translated,
                                                                                    len(translated_sentences)))
        log.info('Sometimes the translator forgets to add the last period.')
        return False
    else:
        log.info('Found {} sentences.'.format(len(original_sentences)))

    data_file = open(config.data_file, 'w', encoding='utf-8')
    data_file.write('{}\n'.format(config.data_file_formatting))
    for i in range(len(original_sentences)):
        data_file.write('{},<p>{}</p>,<p>{}</p>\n'.format(i, original_sentences[i], translated_sentences[i]))

        # Error check.
        if len(original_sentences[i].split(' ')) != len(translated_sentences[i].split(' ')):
            log.warning('Sentence #{} has mismatching number of words:'
                        ' {} and {}.'.format(i, len(original_sentences[i].split(' ')),
                                             len(translated_sentences[i].split(' '))))
            print('Warning: Sentence {} has mismatching number of words.'.format(i))
    data_file.close()

    log.info('.csv data file was successfully stored here: {}.'.format(config.data_file))
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
    original, translated = load_cvs_data()

    _, _ = get_tokenizers(original, translated)


if __name__ == '__main__':
    main()
