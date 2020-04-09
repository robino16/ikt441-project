import config
import urllib.request
import re
import random

log = config.log


def get_filepath(training=True, full=True, merged=False, original=True):
    # todo: move to io_service.py
    # Set training=False to get validation data. 
    # Set full=False to get segmented/augmented data.

    path = 'data/'
    path += 'training/' if training else 'validation/'
    path += 'full-' if full else 'seg_aug-'
    if merged:
        path += 'merged.txt'
    else:
        path += 'original.txt' if original else 'translated.txt'
    return path


def remove_duplicates(l):
    # Returns a list without any duplicates. 
    return list(set(l))


def get_lines_in_file(filepath_in):
    # todo: move to io_service.py
    # Returns a list of all urls stored in data/urls.txt.
    return open(filepath_in, 'r', encoding='utf-8').read().split('\n')


def fetch_html(url_in):
    # Returns a single html object from a url.
    try:
        html = urllib.request.urlopen(url_in)
        return html
    except:
        print('Warning: Failed to fetch html from url: {}'.format(url_in))
        return None


def fetch_all_html_documents(urls_in):
    # Returns a list of html documents as strings.
    print('Debug: Fetching all html documents. Please wait...')
    htmls = []
    for i in range(len(urls_in)):
        print('{}/{}'.format(i + 1, len(urls_in)))
        html = fetch_html(urls_in[i])
        if html is not None:
            htmls.append(decode_html(html))
            html.close()  # Close immediately after decoding so the connection does not close on us.
    return htmls


def decode_html(html_in):
    # Returns a string containing the entire html document.
    return html_in.read().decode('utf-8')


def remove_part_of_string(body_in, from_in, to_in):
    # Removes a sub-section in a string.
    return re.sub('{}.*?{}'.format(from_in, to_in), '', body_in, flags=re.DOTALL)


def filter(body_in):
    # Used to remove Nynorsk/Englush sentences.
    filter_words = [' ein ', ' kva ', ' eg ', ' ikkje ', ' kvifor ', ' no ', ' kven ', ' burda ', 'vanlegvis']
    filter_words += [' heile ', ' verda ', ' nåkre ', ' då ', ' meir ', ' gje ', ' draumar ', ' saman ', ' døydd ']
    filter_words += [' gong ', ' gongen ', ' bu ', ' noreg ', ' kor ']
    filter_words += [' that ', ' why ', ' you ', ' should ', ' not ', ' know ']
    for word in filter_words:
        if body_in.lower().find(word) != -1:
            print('Debug: Filter detected \"{}\" in: \"{}\".'.format(word, body_in))
            return True
    return False


def preserve_punctuation(text_in):
    # Preserves the desired punctuation.
    text_in = text_in.replace('. ', ' . ').replace('? ', ' ? ').replace('! ', ' ! ')
    text_in = text_in.replace(', ', ' , ').replace('  ', ' ')
    return text_in


def parse_html(html_in):
    # Return a list of all valid sentences in a single html document.
    body_texts = re.findall(r'<p>.+?</p>', html_in)
    all_sentences = []
    for body in body_texts:
        body = remove_part_of_string(body, '<', '>')
        body = remove_part_of_string(body, '\(', '\)')
        body = body.replace('- ', '').replace(' ,', ',').replace(' .', '.').replace('  ', ' ')
        body = body.replace('«', '').replace('»', '').replace('– ', '')
        if filter(body):
            return []  # Skip nynorsk/english documents.
        sentences = body.replace('. ', '.$').replace('! ', '!$').replace('? ', '?$').replace(': ', ':$').split('$')
        for sentence in sentences:
            if len(sentence) > config.min_sentence_length:
                sentence = preserve_punctuation(sentence)
                all_sentences.append(sentence)
                # print(len(all_sentences), sentence)
    return all_sentences


def parse_all_html_documents(htmls_in):
    # Return all body text/sentences from all html documents. 
    all_sentences = []
    for html in htmls_in:
        sentences = parse_html(html)
        all_sentences += sentences
    return all_sentences


def augment_sentence(sentence_in):
    # Segments and augments a sentence. 
    words = sentence_in.split(' ')
    aug_seqs = []
    for i in range(len(words) - (config.aug_seq_len - 1)):
        aug_seq = words[i: i + config.aug_seq_len]
        aug_seq = ' '.join(aug_seq)
        aug_seqs.append(aug_seq)
    # todo: Note that it is possible to left/right-shift beginning/end of sentences here
    return aug_seqs


def export_single_file(filename_in, lines_in):
    # Export lines to file. 
    file = open(filename_in, 'w', encoding='utf-8')
    for i in range(len(lines_in)):
        file.write('{}${}\n'.format(i, lines_in[i]))
    file.close()


def export_train_or_test_data(sentences_in, training_data):
    # Change name: 'export_orig_to_dir()'
    # Exports two files: full sentences and segmented version. 
    
    aug_seqs = []  # augmented sequences
    for sentence in sentences_in:
        aug_seqs += augment_sentence(sentence)
    # aug_seqs = remove_duplicates(aug_seqs)  # Disabling this will make the model better at common phrases.

    full_file = get_filepath(training=training_data, full=True, original=True)
    aug_file = get_filepath(training=training_data, full=False, original=True)
    export_single_file(full_file, sentences_in)
    export_single_file(aug_file, aug_seqs)
    return True


def export_original_data(sentences_in):
    # todo: Change name to 'export_orig_train_and_test_data()'
    # Exports original training and original validation data.
    print('Debug: data_generator.py -> export_original_data()')
    print('Remember to train the model from scratch after splitting to training and testing data!')
    # Split into training and validation data
    random.shuffle(sentences_in)
    index = int(len(sentences_in) * config.training_factor)
    export_train_or_test_data(sentences_in[:index], True)
    export_train_or_test_data(sentences_in[index:], False)
    return True


def produce_original_data():
    # A big function handling the entire production of a new dataset using the data/url.txt file. 
    
    print('Debug: data_generator.py -> produce_original_data()')
    
    # Step 1: Get URLs.
    urls = get_lines_in_file(config.url_file)
    urls = remove_duplicates(urls)
    random.shuffle(urls)
    # urls = urls[0:4]  # For debugging purposes we can use less of the urls. 
    
    # Step 2: Fetch all htmls.
    htmls = fetch_all_html_documents(urls)
    if len(htmls) < 1:
        print('Error: No html document was loaded.')
        return False
    
    # Step 3: Extract body text/sentences from all htmls.
    sentences = parse_all_html_documents(htmls)
    if len(sentences) < 1:
        print('Error: Obtained no sentences.')
        return False
    sentences = remove_duplicates(sentences)
    print('Debug: Obtained {} sentences from all html documents.'.format(len(sentences)))
    
    # Step 4: Export the data.
    export_original_data(sentences)
    return True


def get_index_and_body(line_in):
    # Returns the index and body text of a line in a data file. 
    s = line_in.split('$')
    try:
        return int(s[0]), s[1]
    except:
        # print('Debug: Tried to split \"{}\" but ended up with \"{}\".'.format(line_in, s[0]))
        return None, None


def merge_lines(orig, tran):
    # Merge two lines. 
    o_index, o_body = get_index_and_body(orig)
    t_index, t_body = get_index_and_body(tran)
    if o_index is None or t_index is None:
        return None
    if o_index != t_index:  # Check that index in both files are identical. 
        # print('Warning: Failed to merge line {} (original) with line {} (translated).'.format(o_index, t_index))
        return None
    return '{}${}'.format(o_body, t_body)


def merge_full_or_aug(training, full):
    # Merge two files and export the result. 
    f_orig = get_filepath(training=training, full=full, original=True)
    f_tran = get_filepath(training=training, full=full, original=False)

    orig = get_lines_in_file(f_orig)
    tran = get_lines_in_file(f_tran)
    lines = []
    if len(orig) != len(tran):
        print('Warning: {} has {} lines while {} has {} lines.'.format(f_orig, len(orig), f_tran, len(tran)))
    for i in range(min(len(orig), len(tran))):
        s = merge_lines(orig[i], tran[i])
        if s is not None:
            lines.append(s)
        else:
            print('Error: Failed to merge line {} in {} and {}'.format(i, f_orig, f_tran))
    if len(lines) < 1:
        print('Error: No lines where merged.')
        return False
    
    # Export new merged file. 
    f_merg = get_filepath(training=training, full=full, merged=True)
    export_single_file(f_merg, lines)
    print('Debug: Successfully exported {}.'.format(f_merg))
    return True


def merge_dir(training):
    # Merge files in either data/training or data/validation directory.
    merge_full_or_aug(training, True)  # Merge full sentences. 
    merge_full_or_aug(training, False)  # Merge segmented/augmented data.


def merge_all_files():
    # Merge all data files (there are eight in total) resulting in four new merged files. 
    print('data_generator.py -> merge_all_files()')
    merge_dir(True)  # Merge files in training directory.
    merge_dir(False)  # Merge files in validation directory. 


def main():
    app_name = 'IKT441 Project - Dataset Generator'
    print(' --- {} --- '.format(app_name))
    
    # todo: Move merging of data files to it`s own folder. 

    # Operation codes. Pick one operation per run.
    print('Please type the number of the operation you would like to perform and then press enter:')
    print('1. Use url file to produce new dataset.')
    print('2. Create merged file containing both original and translated phrases.')
    opcode = int(input())

    if opcode == 1:
        print('Please remember: Translation must be done manually after this operation has finished.')
        if not produce_original_data():
            print('Error: Failed to produced original data.')
            return -1
    elif opcode == 2:
        merge_all_files()
        print('Finished.')
        return 0
    else:
        print('Error: {} is not a valid input.'.format(opcode))
        return -1


if __name__ == '__main__':
    main()
