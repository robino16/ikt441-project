import config
import model_func
import io_service
import time

log = config.log


def test(model, x, y, tok_ori, tok_tra, f, full=False):
    f += '_full.txt' if full else '.txt'
    model_func.test_model(model, x, y, tok_ori, tok_tra, filename=f, full=full)


def test_model(model, train_x, train_y, test_x, test_y, tok_ori, tok_tra, full=False):
    # Testing on both training and validation data to compare
    test(model, train_x, train_y, tok_ori, tok_tra, 'output/output_train', full=full)
    test(model, test_x, test_y, tok_ori, tok_tra, 'output/output_test', full=full)


def run_app(full=False):
    log.info(config.get_conf())
    train_x, train_y, test_x, test_y, word_cnt_ori, word_cnt_tra, _, tok_ori, tok_tra = io_service.get_all_data(
        full=full)
    model = model_func.create_model(word_cnt_ori, word_cnt_tra, 512)
    if config.train_model:
        start_time = time.time()
        model, history = model_func.train_model(model, train_x, train_y, config.epochs)
        log.info('Finished training after {} seconds.'.format(time.time() - start_time))
        model_func.plot_training(history, config.training_plot_path)

        # Old testing method:
        model_func.test_model(model, test_x, test_y, tok_ori, tok_tra, full=full)

    # New testing method:
    train_x, train_y, test_x, test_y = io_service.get_all_test_data(tok_ori, tok_tra, full=full)
    # model_func.test_model(model, test_x, test_y, tok_ori, tok_tra, filename='output/secondary_test.txt', full=full)
    test_model(model, train_x, train_y, test_x, test_y, tok_ori, tok_tra, full=full)


def main():
    print(' --- {} --- '.format(config.main_app_name))
    log.info(' --- Running application: {} --- '.format(config.main_app_name))

    # Configuration
    full = config.use_full_sentences  # Note: Do not modify this boolean from main.
    config.load_existing_model = True
    config.model_load_file = 'output/' + config.model_folder + '/model.30-0.62.hdf5'
    config.train_model = not config.load_existing_model

    run_app(full)
    log.info("Finished.")


if __name__ == '__main__':
    main()
