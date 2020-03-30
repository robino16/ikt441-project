import config
import model_func
import io_service


log = config.log


def main():
    print(' --- {} --- '.format(config.main_app_name))
    log.info(' --- Running application: {} --- '.format(config.main_app_name))

    # Main application...
    train_x, train_y, test_x, test_y, total_words_original, total_words_translated, max_sequence_length, tokenizer_original, tokenizer_translated = io_service.get_data()

    # Model compilation.
    model = model_func.define_model(total_words_original, total_words_translated, max_sequence_length, 512)

    log.info('Training model for {} epochs.'.format(config.epochs))

    # Train the model.
    # model, history = model_func.train_model(model, train_x, train_y, config.epochs)

    # Plot and save training history
    # model_func.plot_training(history, config.training_plot_path)

    # Test the model.
    model_func.test_model(model, test_x, test_y, max_sequence_length, tokenizer_original, tokenizer_translated)


if __name__ == '__main__':
    main()
