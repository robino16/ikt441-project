import config
import sentence

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import random
import re

# Original texts from: https://helsenorge.no/koronavirus/smitte-og-inkubasjonstid
# The text was translated using this online tool: https://www.apertium.org/index.nob.html?dir=nob-nno#translation

log = config.log

if config.initialize_random_number_generator:
    random.seed(0)


def save_tokenizer(tokenizer_in, filename_in):
    # This function is responsible for saving the tokenizer to a pickle-object so it can be reused.
    with open(filename_in, 'wb') as handle:
        pickle.dump(tokenizer_in, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log.debug('Stored file {} on disk.'.format(filename_in))
    return True


def load_tokenizer(filename_in):
    # Function responsible of loading existing tokenizer file.
    try:
        with open(filename_in, 'rb') as handle:
            tokenizer = pickle.load(handle)
            log.debug('Successfully loaded tokenizer object from file: {}.'.format(filename_in))
            total_words = len(tokenizer.word_index) + 1
            return tokenizer, total_words
    except:
        log.warning('Failed to load tokenizer object from file: {}.'.format(filename_in))
        print('Warning: Failed to load tokenizer object from file {}.'.format(filename_in))
        return None, 0


def get_tokenizer(filename_in, sentences_in):
    # Returns a single tokenizer object.
    tokenizer = Tokenizer(filters=config.tokenizer_filter)
    tokenizer.fit_on_texts(sentences_in)
    # save_tokenizer(tokenizer, filename_in)
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words


def get_tokenizers(original_sentences_in, translated_sentences_in):
    # This function returns the tokenizer objects.
    log.debug('io_service.py -> get_tokenizers()')
    tokenizer_original, _ = get_tokenizer(config.tokenizer_file_original, original_sentences_in)
    tokenizer_translated, _ = get_tokenizer(config.tokenizer_file_translated, translated_sentences_in)
    return tokenizer_original, tokenizer_translated


def get_sentence_objects(original_sentences_in, translated_sentences_in):
    # Returns a list of Sentence objects (containing both original and translated version of a sentence).

    # Error checking.
    if len(original_sentences_in) != len(translated_sentences_in):
        log.error('Number of sentences are not equal: {} (original) '
                  'and {} (translated).'.format(len(original_sentences_in), len(translated_sentences_in)))
        return None

    sentences = []
    for i in range(min(len(original_sentences_in), len(translated_sentences_in))):
        sentences.append(sentence.Sentence(i, original_sentences_in[i], translated_sentences_in[i]))
    return sentences


def objects_to_lists(sentences_in):
    # Function to revert sentence objects to list.
    original = []
    translated = []
    for s in sentences_in:
        original.append(s.original)
        translated.append(s.translated)
    return original, translated


def load_cvs_data():
    log.debug('io_service.py -> load_csv_data()')
    # Function to load data file and format the text so it`s ready to be tokenized.
    original_sentences = []
    translated_sentences = []

    file = open(config.data_file, 'r', encoding='utf-8')
    for line in file:
        # Get original and translated version of the sentence.
        extracted = re.findall(r'<p>.+?</p>', line)

        # Error check. List length should always be 2 (one original and one translated).
        if len(extracted) != 2:
            # log.error('Unexpected number of extracted elements.')  # Should not be possible
            continue

        # Trick to preserve punctuation during training.
        original = extracted[0].replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace('  ', ' ')
        original = original.replace('<p>', '').replace('</p>', '').replace(',', ' ,')

        translated = extracted[1].replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace('  ', ' ')
        translated = translated.replace('<p>', '').replace('</p>', '').replace(',', ' ,')

        original_sentences.append(original)
        translated_sentences.append(translated)
    file.close()
    return original_sentences, translated_sentences


def split_text_to_sentences(text_in):
    # Function to split inputted text into list of sentences (with some quick cleaning).
    sentences = text_in.lower()
    sentences = sentences.replace('\n', '. ').replace('..', '.').replace('  ', ' ')

    # We preserve punctuation.
    # We use $ to split the input into sentences.
    sentences = sentences.replace('.', '.$').replace(':', ':$').replace('?', '?$').replace('!', '!$').replace('$.', '$')
    sentences = sentences.replace('$$', '$').split('$ ')

    sentences_out = []
    for s in sentences:
        if len(s) > 6:  # Skip short sentences.
            sentences_out.append(s.replace('$', ''))  # Remove leftover $ from splitting.

    return sentences_out


def format_original_file():
    log.debug('io_service.py -> format_original_file()')
    # A quick tool to format the original text before utilizing external translator program.
    # The online translator tends to miss punctuations randomly. To avoid this, we need some pre-processing here.
    original_file = open(config.text_file_original, 'r', encoding='utf-8')
    temp = original_file.read()
    original_file.close()

    # We add newline after each sentence. Then we remove double newlines and spaces.
    temp = temp.replace('.', '.\n').replace('\n\n', '\n').replace('\n ', '\n').replace('\n\n', '\n')
    file = open(config.text_file_original, 'w', encoding='utf-8')
    file.write(temp)
    file.close()


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
        log.info('The translator randomly ignores punctuation.')
        print('Info: Mismatch in number of sentences in the data files.')
        return False  # Use pass instead if you want to debug this in data.txt.
    else:
        log.info('Successfully found {} sentences.'.format(len(original_sentences)))
        pass

    data_file = open(config.data_temp_file, 'w', encoding='utf-8')
    # data_file.write('{}\n'.format(config.data_file_formatting))
    for i in range(min(len(translated_sentences), len(original_sentences))):
        data_file.write('{},<p>{}</p>,<p>{}</p>\n'.format(i, original_sentences[i],
                                                          translated_sentences[i]))
    data_file.close()

    log.info('.csv data file was successfully stored here: {}.'.format(config.data_temp_file))
    print('Info: .csv data file was successfully created.')
    return True


def get_max_length(sentences_in, tokenizer_in_original, tokenizer_in_translated):
    # Finds the maximum sentence length of the entire dataset.
    all_sequences = []  # Holds any tokenized sequence.
    for sentence in sentences_in:
        all_sequences.append(tokenizer_in_original.texts_to_sequences([sentence.original])[0])
        all_sequences.append(tokenizer_in_translated.texts_to_sequences([sentence.translated])[0])

    max_length = max([len(x) for x in all_sequences])
    print('Debug: Max sequence length: {}.'.format(max_length))
    log.debug('max_sequence_length={}'.format(max_length))
    return max_length


def split_seq_into_segments(seq_in):
    # Convert a long sequences into segments which can be inputted into our model.
    max_len = len(seq_in)
    length = config.max_sequence_length
    segments = []

    seq_b = seq_in.copy()  # Holds a copy of entire original sequence
    seq_e = seq_in.copy()

    # Appends right shifted version of beginning.
    temp = []
    for i in range(length - 1):
        seq_b = right_shift_fill_zeros(seq_b)
        temp.append(seq_b)
    segments += temp[::-1]

    for i in range(max_len - (length - 1)):
        seg = seq_in[i: i + length]
        segments.append(seg)

    # Appends left shifted version of end.
    for i in range(length - 1):
        seq_e = left_shift_fill_zeros(seq_e)
        segments.append(seq_b)
    return segments


def right_shift_fill_zeros(array_in):
    # [54, 23, 23, 554] becomes -> [0, 54, 23, 23]
    # For the first part of a sentence.
    return [0] + array_in[:-1]


def left_shift_fill_zeros(array_in):
    # [54, 23, 23, 554] becomes -> [23, 23, 554, 0]
    # For the last part of a sentence.
    return array_in[1:] + [0]


def get_training_and_testing_data(sentences_in, length_in, tokenizer_original, tokenizer_translated):
    print('Debug: Augmenting all input sequences.')

    # Augment all sequences.
    augmented_sequences = []
    for sentence in sentences_in:
        original = tokenizer_original.texts_to_sequences([sentence.original])[0]
        translated = tokenizer_translated.texts_to_sequences([sentence.translated])[0]
        max_len = min(len(original), len(translated))

        if max_len < length_in:
            continue  # Skip short sequences.

        orig_b = original.copy()  # Holds a copy of entire original sequence
        orig_e = original.copy()
        tran_b = translated.copy()  # Holds a copy of entire translated sequence
        tran_e = translated.copy()
        for i in range(length_in - 1):
            # Beginning of sentence:
            orig_b = right_shift_fill_zeros(orig_b)
            tran_b = right_shift_fill_zeros(tran_b)
            temp_b = orig_b[:length_in] + tran_b[:length_in]
            augmented_sequences.append(temp_b)

            # End of sentence:
            orig_e = left_shift_fill_zeros(orig_e)
            tran_e = left_shift_fill_zeros(tran_e)
            temp_e = orig_e[-length_in:] + tran_e[-length_in:]
            augmented_sequences.append(temp_e)

        for i in range(max_len - (length_in - 1)):
            orig = original[i: i + length_in]
            tran = translated[i: i + length_in]
            temp = orig + tran
            augmented_sequences.append(temp)
            # continue  # Skip augmentation.

            # Extra augmentation by reversing.
            # temp_reversed = orig[::-1] + tran[::-1]
            # augmented_sequences.append(temp_reversed)

    augmented_sequences = pad_sequences(augmented_sequences, maxlen=length_in * 2, padding='pre')

    # Split to training and testing.
    random.shuffle(augmented_sequences)
    max_nr_of_sequences = config.max_nr_of_training_seqs
    if max_nr_of_sequences > len(augmented_sequences):
        augmented_sequences = augmented_sequences[:max_nr_of_sequences]
    index = int(len(augmented_sequences) * config.training_factor)
    train_x, train_y = augmented_sequences[:index, :length_in], augmented_sequences[:index, length_in:]
    test_x, test_y = augmented_sequences[index:, :length_in], augmented_sequences[index:, length_in:]

    # Log/print results.
    log.info('{} sequences will be used for training.'.format(len(train_y)))
    print('Info: {} training sequences.'.format(len(train_y)))
    log.info('{} sequences will be used for testing.'.format(len(test_y)))
    print('Info: {} testing sequences.'.format(len(test_y)))

    return train_x, train_y, test_x, test_y


def get_data():
    # This function returns the training and testing data.
    log.debug('io_service.py -> get_data()')

    # Load the data.
    # create_csv_data_file()  # Use this if we want to create a new dataset.
    original_sentences, translated_sentences = load_cvs_data()

    # Convert to sentence objects.
    sentences = get_sentence_objects(original_sentences, translated_sentences)
    random.shuffle(sentences)

    # Load the tokenizer objects.
    tokenizer_original, tokenizer_translated = get_tokenizers(original_sentences, translated_sentences)

    # Get max sequence length
    # max_sequence_length = get_max_length(sentences, tokenizer_original, tokenizer_translated)
    max_sequence_length = config.max_sequence_length

    train_x, train_y, test_x, test_y = get_training_and_testing_data(sentences,
                                                                     max_sequence_length,
                                                                     tokenizer_original,
                                                                     tokenizer_translated)

    # Get max number of words (for input + output).
    # Note: This function may also return these values.
    total_words_original = len(tokenizer_original.word_index) + 1
    total_words_translated = len(tokenizer_translated.word_index) + 1

    print('Info: Total words in original text: {}.'.format(total_words_original))
    log.info('Total words in original text: {}.'.format(total_words_original))

    print('Info: Total words in translated text: {}.'.format(total_words_translated))
    log.info('Total words in translated text: {}.'.format(total_words_translated))

    return train_x, train_y, test_x, test_y, total_words_original, total_words_translated, max_sequence_length, tokenizer_original, tokenizer_translated


def main():
    print(' --- {} --- '.format(config.io_service_app_name))
    log.info(' --- Running application: {} --- '.format(config.io_service_app_name))

    # The main function displays how we can use the io_service from an external application.

    # format_original_file()

    # Create new dataset
    create_new_dataset = False
    if create_new_dataset:
        if not create_csv_data_file():
            print('Error: Failed to generate new dataset. See log output for more info.')
            log.error('Failed to generate new dataset.')

    # Get training and testing data
    train_x, train_y, test_x, test_y, total_words_original, total_words_translated, _, _, _ = get_data()


if __name__ == '__main__':
    main()
