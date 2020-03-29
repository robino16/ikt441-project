import config

log = config.log()


def main():
    print(' --- {} --- '.format(config.main_app_name))
    log.info(' --- Running application: {} --- '.format(config.main_app_name))

    # Main application...


if __name__ == '__main__':
    main()
