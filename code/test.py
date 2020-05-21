import config
import model_func
import io_service

log = config.log


def main():
    print(' --- {} --- '.format(config.test_app_name))
    log.info(' --- Running application: {} --- '.format(config.test_app_name))

    config.load_existing_model = True
    config.model_load_file = 'output/models/model.15-0.59.hdf5'

    # Get data.
    train_x, train_y, test_x, test_y, word_count_ori, word_count_tra, _, tok_ori, tok_tra = io_service.get_all_data()

    # Create model.
    model = model_func.create_model(word_count_ori, word_count_tra, 512)

    train_x, train_y, test_x, test_y = io_service.get_all_test_data(tok_ori, tok_tra)

    # Test the model.
    model_func.test_model(model, train_x, train_y, tok_ori, tok_tra, filename='output/output_train.txt')
    model_func.test_model(model, test_x, test_y, tok_ori, tok_tra, filename='output/output_test.txt')

    log.info("Finished.")


if __name__ == '__main__':
    main()
