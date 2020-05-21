# Set the matplotlib backend so figures can be saved in the background.
import matplotlib
matplotlib.use("Agg")

import config
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, RepeatVector
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

log = config.log


def create_model(in_vocab, out_vocab, units):
    # Build the model.
    log.debug('Building model...')

    model = None

    if not config.load_existing_model:
        model = Sequential()
        model.add(Embedding(in_vocab, units, input_length=config.input_sequence_length, mask_zero=True))
        model.add(LSTM(units))
        model.add(RepeatVector(config.output_sequence_length))
        model.add(LSTM(units, return_sequences=True))
        model.add(Dense(out_vocab, activation='softmax'))

        # Compile model.
        log.debug('Compiling model...')

        rms = optimizers.RMSprop(lr=0.001)
        model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

    # Load model.
    if config.load_existing_model:
        try:
            model = load_model(config.model_load_file)
            log.info('Successfully loaded model from file: \"{}\"'.format(config.model_load_file))
        except:
            print('Error: Failed to load existing model.')
            log.warning('Failed to load existing model from file: \"{}\"'.format(config.model_load_file))
            pass

    return model


def train_model(model, train_x, train_y, epochs):
    # Train the model.
    log.debug('Training model for {} epochs...'.format(epochs))
    f = 'output/models/finished_model.hdf5'

    if not config.use_full_sentences:
        checkpoint = ModelCheckpoint(config.model_save_file, monitor='val_loss', verbose=1)
        history = model.fit(train_x, train_y.reshape(train_y.shape[0], train_y.shape[1], 1), epochs=epochs,
                            batch_size=config.batch_size, validation_split=config.validation_split, verbose=1,
                            callbacks=[checkpoint])
    else:
        history = model.fit(train_x, train_y.reshape(train_y.shape[0], train_y.shape[1], 1), epochs=epochs,
                            batch_size=config.batch_size, validation_split=config.validation_split, verbose=1)
        f = 'output/models_full/finished_model.hdf5'
    model.save(f)

    return model, history


def test_model(model, test_x, test_y, tok_ori, tok_tra, filename=config.output_file, full=False):
    # Test the model.
    log.info('Testing model...')

    # Get predicted translations from trained model.
    if full:
        preds = model.predict_classes(test_x.reshape((test_x.shape[0], test_x.shape[1])))
    else:
        preds = [model.predict_classes(instance) for instance in test_x]

    # Convert integer sequences to texts.
    original_bokmaal_sentences = convert_text(test_x, tok_ori, full=full)
    original_nynorsk_sentences = convert_text(test_y, tok_tra, full=full)
    preds_text = convert_text(preds, tok_tra, full=full)

    # Open file for printing predicted translations to.
    file = open(filename, 'w', encoding='utf-8')
    file.write(config.get_conf())

    # Write original bokmaal, nynorsk and predicted translated sentences to file.
    for i in range(len(preds_text)):
        file.write('\nBokmål:\n{}'.format(original_bokmaal_sentences[i]))
        file.write('\nNynorsk:\n{}'.format(original_nynorsk_sentences[i]))
        file.write('\nGenerated nynorsk:\n{}\n'.format(preds_text[i]))

    file.close()
    log.info('Generated nynorsk sentences was successfully saved to file: \"{}\".'.format(filename))
    print('Generated nynorsk sentences was successfully saved to file: \"{}\".'.format(filename))


def convert_text(seqs, tokenizer, full):
    if full:
        return convert_full(seqs, tokenizer)
    return [convert_sequences_into_text(instance, tokenizer) for instance in seqs]


def convert_full(seqs, tokenizer):
    # Function for converting integer sequences into a text.
    preds_text = []
    for i in seqs:
        temp = []
        for j in range(len(i)):
            t = get_word(i[j], tokenizer)
            if j > 0:
                if (t == get_word(i[j - 1], tokenizer)) or (t is None):
                    temp.append('$')
                else:
                    temp.append(t)
            else:
                if t is None:
                    temp.append('$')
                else:
                    temp.append(t)
        text = ' '.join(temp)
        text = text.capitalize().replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').rstrip()
        preds_text.append(text)
    return preds_text


def convert_sequences_into_text(sequences, tokenizer):
    # Function for converting integer sequences into a text.
    text = ''

    for s in sequences:
        for i in range(len(s)):
            t = get_word(s[i], tokenizer)

            if i > 0:
                if (t == get_word(s[i - 1], tokenizer)) or (t is None):
                    text += '$ '
                else:
                    text += t + ' '
            else:
                if t is not None and t != ' ':
                    text += t + ' '
                else:
                    text += '$ '

    # Capitalize the first word of the sentence, fix punctuation issues and remove trailing whitespace
    text = text.capitalize().replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').rstrip()

    return text


def get_word(n, tokenizer):
    #  Convert integer in sequence to corresponding word.
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


def plot_training(history, plot_path):
    # Construct a plot that plots and saves the training history.
    log.info('Plotting and saving training history...')

    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend(loc='upper right')
    plt.title("Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.savefig(plot_path)


def main():
    # For testing purposes.
    pass


if __name__ == '__main__':
    main()
