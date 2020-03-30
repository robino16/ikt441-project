import config
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, RepeatVector

log = config.log

def define_model(in_vocab, out_vocab, in_timesteps, out_timesteps,units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model


def main():
    pass


if __name__ == '__main__':
    main()