# Set the matplotlib backend so figures can be saved in the background.
import matplotlib
matplotlib.use("Agg")

import config
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, RepeatVector
from keras import optimizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

log = config.log


def create_model(in_vocab, out_vocab, time_steps, units):
    # Build the model.
    log.debug('Building model...')

    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=time_steps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(time_steps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))

    # Compile model.
    log.debug('Compiling model...')

    rms = optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

    # Load weights.
    if config.load_existing_weights:
        try:
            model.load_weights(config.weights_file)
            log.info('Successfully loaded model weights from file: \"{}\"'.format(config.weights_file))
        except:
            print('Error: Failed to load existing weights.')
            log.warning('Failed to load existing weights from file: \"{}\"'.format(config.weights_file))
            pass

    return model


def train_model(model, train_x, train_y, epochs):
    # Train the model.
    log.debug('Training model for {} epochs...'.format(epochs))

    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')

    history = model.fit(train_x, train_y.reshape(train_y.shape[0], train_y.shape[1], 1), epochs=epochs,
                        batch_size=config.batch_size, validation_split=config.validation_split, verbose=1,
                        callbacks=[early_stop])

    if config.save_weights:
        try:
            model.save_weights(config.weights_file)
            log.debug('Saved model weights to file: \"{}\".'.format(config.weights_file))
            print('Saved model weights to file: \"{}\".'.format(config.weights_file))
        except:
            print('Error: Failed to save weights after training.')
            log.error('Failed to save weights after training.')

    return model, history


def test_model(model, test_x, test_y, tokenizer_original, tokenizer_translated):
    # Test the model.
    log.info('Testing model...')

    # Get predicted translations from trained model.
    preds = model.predict_classes(test_x.reshape((test_x.shape[0], test_x.shape[1])))

    # Convert integer sequences to texts.
    original_bokmaal_sentences = convert_sequences_into_texts(test_x, tokenizer_original)
    original_nynorsk_sentences = convert_sequences_into_texts(test_y, tokenizer_translated)
    preds_text = convert_sequences_into_texts(preds, tokenizer_translated)

    # Open file for printing predicted translations to.
    file = open(config.output_file, 'w', encoding='utf-8')
    file.write(get_conf())

    # Write original bokmaal, nynorsk and predicted translated sentences to file.
    for i in range(len(preds_text)):
        file.write('\nBokmål:\n{}'.format(original_bokmaal_sentences[i]))
        file.write('\nNynorsk:\n{}'.format(original_nynorsk_sentences[i]))
        file.write('\nGenerated nynorsk:\n{}\n'.format(preds_text[i]))

    file.close()
    log.info('Generated nynorsk sentences was successfully saved to file: \"{}\".'.format(config.output_file))
    print('Generated nynorsk sentences was successfully saved to file: \"{}\".'.format(config.output_file))


def get_conf():
    conf = 'Configuration:\n' \
           'Training epochs: {} epochs.\n' \
           'Training factor: {}.\n'.format(config.epochs,
                                           config.training_factor)

    return conf


def convert_sequences_into_texts(sequences, tokenizer):
    # Function for converting integer sequences into texts.
    texts = []

    for s in sequences:
        temp = []

        for i in range(len(s)):
            t = get_word(s[i], tokenizer)

            if i > 0:
                if (t == get_word(s[i - 1], tokenizer)) or (t is None):
                    pass
                else:
                    temp.append(t + ' ')
            else:
                if t is not None and t != ' ':
                    temp.append(t + ' ')

        # Fix punctuation issues.
        temp = ''.join(temp)
        temp = temp.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        temp = temp[:1].upper() + temp[1:]  # Capitalize first letter.
        texts.append(temp)

    return texts


def get_word(n, tokenizer):
    #  Convert integer in sequence to corresponding word.
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


def plot_training(history, plot_path):
    # Construct a plot that plots and saves the training history.
    log.info('Plotting and saving training history...')

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend(loc='upper right')
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.savefig(plot_path)


def main():
    # For testing purposes.
    pass


if __name__ == '__main__':
    main()
