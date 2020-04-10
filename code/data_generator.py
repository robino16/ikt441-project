import config
import urllib.request
import re
import random
import io_service
import time

log = config.log


def remove_duplicates(l):
    # Returns a list without any duplicates. 
    return list(set(l))


def fetch_html(url_in):
    # Returns a single html object from a url.
    try:
        html = urllib.request.urlopen(url_in)
        return html
    except:
        print('Warning: Failed to fetch html from url: {}'.format(url_in))
        return None


def show_progress(t_elapsed, current_step, total_steps):
    if current_step < 1:
        return False
    percentage = 100 * (current_step/total_steps)
    t_remaining = (t_elapsed / current_step) * (total_steps - current_step) if current_step > 0 else None
    t_remaining = '{}'.format('%.0f' % t_remaining)
    print('Progress: {}/{} ({}%), Time remaining: {}s.'.format(current_step, total_steps, '%.0f' % percentage, t_remaining))
    return True


def fetch_all_html_documents(urls_in):
    # Returns a list of html documents as strings.
    print('Debug: Fetching all html documents. Please wait...')
    start_time = time.time()
    htmls = []
    for i in range(len(urls_in)):
        html = fetch_html(urls_in[i])
        if html is not None:
            htmls.append(decode_html(html))
            html.close()  # Close immediately after decoding so the connection does not close on us.
        elapsed_time = time.time() - start_time
        show_progress(elapsed_time, i + 1, len(urls_in))
    return htmls


def decode_html(html_in):
    # Returns a string containing the entire html document.
    return html_in.read().decode('utf-8')


def remove_part_of_string(body_in, from_in, to_in):
    # Removes a sub-section in a string.
    return re.sub('{}.*?{}'.format(from_in, to_in), '', body_in, flags=re.DOTALL)


def filter_words(body_in):
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
        if filter_words(body):
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


def segment_sentence(sentence_in):
    # Segments a sentence.
    words = sentence_in.split(' ')
    aug_seqs = []
    for i in range(len(words) - (config.aug_seq_len - 1)):
        aug_seq = words[i: i + config.aug_seq_len]
        aug_seq = ' '.join(aug_seq)
        if i == 0:  # Start of sentence.
            aug_seq += '$0'
        elif i == len(words) - (config.aug_seq_len - 1) - 1:  # End of sentence.
            aug_seq += '$2'
        else:  # Middle of sentence.
            aug_seq += '$1'
        aug_seqs.append(aug_seq)
    return aug_seqs


def export_orig_to_dir(sentences_in, training_data):
    # Exports two files: full sentences and segmented version. 
    
    aug_seqs = []  # augmented sentences
    full_seqs = []  # original sentences
    for sentence in sentences_in:
        aug_seqs += segment_sentence(sentence)
        full_seqs.append(sentence + '$3')
    # aug_seqs = remove_duplicates(aug_seqs)  # Disabling this will make the model better at common phrases.

    random.shuffle(full_seqs)
    random.shuffle(aug_seqs)

    full_file = io_service.get_filepath(training=training_data, full=True, original=True)
    aug_file = io_service.get_filepath(training=training_data, full=False, original=True)
    io_service.export_lines_to_file(full_file, full_seqs)
    io_service.export_lines_to_file(aug_file, aug_seqs)
    return True


def export_orig_train_and_test(sentences_in):
    # Exports original training and original validation data.
    print('Debug: data_generator.py -> export_original_data()')
    print('Remember to train the model from scratch after splitting to training and testing data!')
    # Split into training and validation data
    random.shuffle(sentences_in)
    index = int(len(sentences_in) * config.training_factor)
    export_orig_to_dir(sentences_in[:index], True)
    export_orig_to_dir(sentences_in[index:], False)
    return True


def produce_original_data():
    # A big function handling the entire production of a new dataset using the data/url.txt file. 
    
    print('Debug: data_generator.py -> produce_original_data()')
    
    # Step 1: Get URLs.
    urls = io_service.get_lines_in_file(config.url_file)
    urls = remove_duplicates(urls)
    random.shuffle(urls)
    urls = urls[0:104]  # For debugging purposes we can use less of the urls.
    
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
    export_orig_train_and_test(sentences)
    return True


def main():
    app_name = 'IKT441 Project - Dataset Generator'
    print(' --- {} --- '.format(app_name))

    if not produce_original_data():
        print('Error: Failed to produced original data.')
        return -1
    return 0


if __name__ == '__main__':
    main()
