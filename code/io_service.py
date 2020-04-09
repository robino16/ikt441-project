import config
from keras.preprocessing.text import Tokenizer
import data_generator
from keras.preprocessing.sequence import pad_sequences
import random

log = config.log


def get_filepath(training=True, full=True, merged=False, original=True):
    # Returns data file paths.
    path = 'data/'
    path += 'training/' if training else 'validation/'
    path += 'full-' if full else 'seg_aug-'
    if merged:
        path += 'merged.txt'
    else:
        path += 'original.txt' if original else 'translated.txt'
    return path


def get_lines_in_file(filepath_in):
    # Returns a list of all urls stored in data/urls.txt.
    return open(filepath_in, 'r', encoding='utf-8').read().split('\n')


def export_lines_to_file(filename_in, lines_in):
    # Export lines to file.
    file = open(filename_in, 'w', encoding='utf-8')
    for i in range(len(lines_in)):
        file.write('{}${}\n'.format(i, lines_in[i]))
    file.close()


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
    lines = get_lines_in_file(filepath_in)
    random.shuffle(lines)
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
    f_train = get_filepath(training=True, full=full, merged=True)
    f_test = get_filepath(training=False, full=full, merged=True)
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
    # Returns input and output sequences.
    f_train, f_test = get_filenames(full=not segmented)
    f = f_train if training else f_test
    orig_phrases, tran_phrases = load_merged_data(f)
    index = config.max_nr_of_training_seqs if training else config.max_nr_of_testing_seqs
    orig = tokenize_and_pad_sentences(orig_phrases[:index], tokenizer_original)
    tran = tokenize_and_pad_sentences(tran_phrases[:index], tokenizer_translated)
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

    train_x, train_y, test_x, test_y, word_count_ori, word_count_tra, max_seq_len, tok_ori, tok_tra = get_all_data()

    # Print results
    print('Debug: {} training sequences.'.format(len(train_y)))
    print('Debug: {} validation sequences.'.format(len(test_y)))
    print('Debug: Word count original: {}.'.format(word_count_ori))
    print('Debug: Word count translated: {}.'.format(word_count_tra))
    print('Debug: Max sequence length: {}.'.format(max_seq_len))


if __name__ == '__main__':
    main()
