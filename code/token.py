from keras.preprocessing.text import Tokenizer
import pickle

import config

# log = config.log


def save_tokenizer(tokenizer_in, filename_in):
    # This function is responsible for saving the tokenizer to a pickle-object so it can be reused.
    with open(filename_in, 'wb') as handle:
        pickle.dump(tokenizer_in, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(filename_in):
    # Function responsible of loading existing tokenizer file.
    try:
        with open(filename_in, 'rb') as handle:
            tokenizer = pickle.load(handle)
            # log.debug('Successfully loaded tokenizer object from file: {}.'.format(filename_in))
            return tokenizer
    except:
        # log.warning('Failed to load tokenizer object from file: {}.'.format(filename_in))
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
    # log.debug('Tokenizer ({}) has {} indexed words.'.format(filename_in, total_words))
    return tokenizer


def get_tokenizers(original_sentences_in, translated_sentences_in):
    # This function returns the tokenizer objects.
    # Note: There are two tokenizer objects, one for the original language and one for the translation.
    tokenizer_original = get_tokenizer(config.tokenizer_file_original, original_sentences_in)
    tokenizer_translated = get_tokenizer(config.tokenizer_file_translated, translated_sentences_in)
    return tokenizer_original, tokenizer_translated
