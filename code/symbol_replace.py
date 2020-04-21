import config
import io_service


def replace_symbol(filename_in, symbol_a, symbol_b):
    print(filename_in)
    lines = io_service.get_lines_in_file(filename_in)
    temp = []
    for line in lines:
        t = line.split('$')[1:]
        t = '$'.join(t)
        t = t.replace(symbol_a, symbol_b)
        temp.append(t)
    io_service.export_lines_to_file(filename_in, temp)
    print('done')


def replace_in_merged_files(symbol_a, symbol_b):
    replace_symbol(io_service.get_filepath(training=True, full=False, merged=True), symbol_a, symbol_b)
    replace_symbol(io_service.get_filepath(training=False, full=False, merged=True), symbol_a, symbol_b)


def main():
    previous_symbol = 'EMPTY'
    replace_in_merged_files(previous_symbol, config.empty_word)


if __name__ == '__main__':
    main()
