import config
import model_func
import io_service
import time

log = config.log


def main():
    print(' --- {} --- '.format(config.main_app_name))
    log.info(' --- Running application: {} --- '.format(config.main_app_name))

    start_time = time.time()

    log.info(model_func.get_conf())

    # Get data.
    train_x, train_y, test_x, test_y, total_words_original, total_words_translated, max_sequence_length, tokenizer_original, tokenizer_translated = io_service.get_data()

    # Create model.
    model = model_func.create_model(total_words_original, total_words_translated, max_sequence_length, 512)

    if config.train_weights:
        # Train the model.
        model, history = model_func.train_model(model, train_x, train_y, config.epochs)

        # Plot and save training history
        model_func.plot_training(history, config.training_plot_path)

    # Test the model.
    model_func.test_model(model, test_x, test_y, tokenizer_original, tokenizer_translated)

    log.info("Finished after {} seconds.\n".format('%.0f' % (time.time() - start_time)))


if __name__ == '__main__':
    main()
