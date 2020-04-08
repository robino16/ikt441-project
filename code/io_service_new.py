import config
from keras.preprocessing.text import Tokenizer
import data_generator
from keras.preprocessing.sequence import pad_sequences

log = config.log


def get_total_words(tokenizer_in):
    return len(tokenizer_in.word_index) + 1


def get_tokenizer(sentences_in):
    # Returns a single tokenizer object.
    tokenizer = Tokenizer(filters=config.tokenizer_filter)
    tokenizer.fit_on_texts(sentences_in)
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words


def get_tokenizers():
    # Returns original and translated tokenizer objects.
    original_sentences, translated_sentences = get_both_train_and_test_sentences()
    tok_ori, _ = get_tokenizer(original_sentences)
    tok_tra, _ = get_tokenizer(translated_sentences)
    return tok_ori, tok_tra


def format_sentence(sentence_in):
    return sentence_in.lower()


def load_merged_data(filepath_in):
    # Returns original sentences and translated sentences in the file.
    lines = data_generator.get_lines_in_file(filepath_in)
    orig, tran = [], []  # Original and translated sentences.
    for i in range(len(lines)):
        s = lines[i].split('$')
        if len(s) != 3:
            print('Warning: Could not parse line {} in file {}: {}.'.format(i, filepath_in, lines[i]))
            continue
        orig.append(format_sentence(s[1]))
        tran.append(format_sentence(s[2]))
    return orig, tran


def get_filenames(full=True):
    f_train = data_generator.get_filepath(training=True, full=full, merged=True)
    f_test = data_generator.get_filepath(training=False, full=full, merged=True)
    return f_train, f_test


def get_both_train_and_test_sentences():
    # Used by tokenizer to tokenize all possible words in our entire dataset.
    f_train, f_test = get_filenames()
    train_orig, train_tran = load_merged_data(f_train)
    test_orig, test_tran = load_merged_data(f_test)
    return train_orig + test_orig, train_tran + test_tran


def tokenize_and_pad_sentences(sentences_in, tokenizer_in):
    sentences_in = tokenizer_in.texts_to_sequences(sentences_in)
    sentences_in = pad_sequences(sentences_in, maxlen=config.max_sequence_length, padding='post')
    return sentences_in


def get_data(tokenizer_original, tokenizer_translated, training=True, segmented=True):
    # Loads training data by default.
    # Load validation by setting training=True.
    # Optionally use segmentation in validation by setting segmented=True.
    f_train, f_test = get_filenames(full=not segmented)
    f = f_train if training else f_test
    orig_phrases, tran_phrases = load_merged_data(f)
    orig = tokenize_and_pad_sentences(orig_phrases, tokenizer_original)
    tran = tokenize_and_pad_sentences(tran_phrases, tokenizer_translated)
    return orig, tran


def get_all_data():
    print('io_service.py (New version) -> get_all_data()')

    # Tokenizer
    tok_ori, tok_tra = get_tokenizers()
    word_count_ori = get_total_words(tok_ori)
    word_count_tra = get_total_words(tok_tra)

    # Training and validation data
    train_x, train_y = get_data(tok_ori, tok_tra)
    test_x, test_y = get_data(tok_ori, tok_tra, training=False, segmented=True)

    max_seq_len = config.max_sequence_length
    return train_x, train_y, test_x, test_y, word_count_ori, word_count_tra, max_seq_len, tok_ori, tok_tra


def main():
    print(' --- {} (New version) ---'.format(config.io_service_app_name))
    log.info(' --- Running application: {} --- '.format(config.io_service_app_name))

    a, b, c, d, e, f, g, h, i = get_all_data()

    # Print results
    print('Info: {} sequences for training.'.format(len(b)))
    print('Info: {} sequences for validation.'.format(len(c)))
    print('Info: Word count original: {}.'.format(e))
    print('Info: Word count translated: {}.'.format(f))
    print('Info: Max sequence length: {}.'.format(g))


if __name__ == '__main__':
    main()
