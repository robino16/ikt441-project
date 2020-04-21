import config
import model_func
import io_service

log = config.log


def main():
    print(' --- {} --- '.format(config.main_app_name))
    log.info(' --- Running application: {} --- '.format(config.main_app_name))

    log.info(model_func.get_conf())

    # Get data.
    train_x, train_y, test_x, test_y, word_count_ori, word_count_tra, max_seq_len, tok_ori, tok_tra = io_service.get_all_data()

    # Create model.
    model = model_func.create_model(word_count_ori, word_count_tra, max_seq_len, 512)

    if config.train_model:
        # Train the model.
        model, history = model_func.train_model(model, train_x, train_y, config.epochs)

        # Plot and save training history
        model_func.plot_training(history, config.training_plot_path)

    # Test the model.
    model_func.test_model(model, test_x, test_y, tok_ori, tok_tra)

    log.info("Finished.")


if __name__ == '__main__':
    main()
