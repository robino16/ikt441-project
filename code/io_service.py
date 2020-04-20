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
    orig, tran, section = [], [], []  # Original and translated sentences.
    for i in range(len(lines)):
        s = lines[i].split('$')
        if len(s) != 4:
            # print('Warning: Could not parse line {} in file {}: \"{}\".'.format(i, filepath_in, lines[i]))
            continue
        orig.append(format_sentence(s[1]))
        tran.append(format_sentence(s[2]))
        section.append(int(s[3]))
    return orig, tran, section


def get_filenames(full=False):
    f_train = get_filepath(training=True, full=full, merged=True)
    f_test = get_filepath(training=False, full=full, merged=True)
    return f_train, f_test


def get_both_train_and_test_sentences():
    # Used by tokenizer to tokenize all possible words in our entire dataset.
    f_train, f_test = get_filenames()
    train_orig, train_tran, _ = load_merged_data(f_train)
    test_orig, test_tran, _ = load_merged_data(f_test)
    return train_orig + test_orig, train_tran + test_tran


def pad_zeros(seq_in, maxlen=config.aug_seq_len, padding='post'):
    temp = seq_in
    for i in range(maxlen - len(seq_in)):
        if padding == 'pre':
            temp = [0] + temp
        elif padding == 'post':
            temp = temp + [0]
    if maxlen - len(seq_in) < 0 and padding == 'none':
        temp = temp[:maxlen]
    return temp


def split_seq_to_segments(seq_in, increment_by_one=False, aug=False):
    if len(seq_in) <= config.aug_seq_len:
        return pad_sequences([seq_in], maxlen=config.max_sequence_length, padding='post')
    if not increment_by_one:
        aug = False  # We cannot have augmentation here.
    segments = []
    sections = []
    lim = len(seq_in) - (config.aug_seq_len - 1) if increment_by_one else len(seq_in)
    i = 0
    while i < lim:
        seg = seq_in[i: i + config.aug_seq_len]
        segments.append(pad_zeros(seg))

        i += 1 if increment_by_one else config.aug_seq_len
        if i == 1:
            sections.append(0)
        elif i >= lim:
            sections.append(2)
        else:
            sections.append(1)
    if aug:
        segments = augmentation(segments, sections)
    segments = pad_sequences(segments, maxlen=config.max_sequence_length, padding='post')
    return segments  # note: We can optionally return segments[0]. Depends on how we use our model during testing.


def right_shift_fill(array_in, fill=0):
    return [fill] + array_in[:-1]


def left_shift_fill(array_in, fill=0):
    return array_in[1:] + [fill]


def augmentation(sentences_in, sections_in):
    augmented_sequences = []
    for i in range(min(len(sentences_in), len(sections_in))):
        if sections_in[i] == 0:
            temp = sentences_in[i].copy()
            temp_list = [temp]
            for j in range(config.aug_seq_len - 1):
                temp = right_shift_fill(temp)
                temp_list.append(pad_zeros(temp, maxlen=config.aug_seq_len, padding='none'))
            augmented_sequences += temp_list[::-1]
        elif sections_in[i] == 2:
            temp = sentences_in[i].copy()
            augmented_sequences.append(temp)
            for j in range(config.aug_seq_len - 1):
                temp = left_shift_fill(temp)
                augmented_sequences.append(pad_zeros(temp, maxlen=config.aug_seq_len, padding='none'))
        else:
            augmented_sequences.append(sentences_in[i])
    return augmented_sequences


def replace_empty_words(sentences_in, empty_token):
    sentences = []
    for s in sentences_in:
        temp = []
        for w in s:
            if w == empty_token:
                temp.append(0)
            else:
                temp.append(w)
        sentences.append(temp)
    return sentences


def tokenize_and_pad_sentences(sentences_in, sections_in, tokenizer_in, segmented=True):
    empty_token = tokenizer_in.texts_to_sequences([config.empty_word])[0][0]
    sentences_in = tokenizer_in.texts_to_sequences(sentences_in)
    sentences_in = replace_empty_words(sentences_in, empty_token)
    if segmented:
        sentences_in = pad_sequences(sentences_in, maxlen=config.max_sequence_length, padding='post')
    return sentences_in


def get_data(tokenizer_original, tokenizer_translated, training=True, segmented=True):
    # Returns input and output sequences.
    f_train, f_test = get_filenames(full=not segmented)
    f = f_train if training else f_test
    orig_phrases, tran_phrases, sections = load_merged_data(f)
    # orig_phrases, tran_phrases, sections = ['Han må iverksette en'], ['Han må setja i verk ein'], [2]
    index = config.max_nr_of_training_seqs if training else config.max_nr_of_testing_seqs
    orig = tokenize_and_pad_sentences(orig_phrases[:index], sections[:index], tokenizer_original, segmented=segmented)
    tran = tokenize_and_pad_sentences(tran_phrases[:index], sections[:index], tokenizer_translated, segmented=segmented)
    return orig, tran


def get_all_data():
    print('io_service.py (New version) -> get_all_data()')

    # Tokenizer
    tok_ori, tok_tra = get_tokenizers()
    word_count_ori = get_total_words(tok_ori)
    word_count_tra = get_total_words(tok_tra)

    # Training and validation data
    train_x, train_y = get_data(tok_ori, tok_tra)
    test_x, test_y = get_data(tok_ori, tok_tra, training=False, segmented=False)

    test_x = [split_seq_to_segments(seq) for seq in test_x]
    test_y = [split_seq_to_segments(seq) for seq in test_y]

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

    validation_data = get_data(tok_ori, tok_tra, training=False, segmented=False)
    print('validation_data[0][0]={}'.format(validation_data[0][0]))
    print('segmented=\n{}'.format(split_seq_to_segments(validation_data[0][0], increment_by_one=True, aug=True)))


if __name__ == '__main__':
    main()
