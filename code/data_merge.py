import config
import io_service

log = config.log


def get_index_and_body(line_in):
    # Returns the index and body text of a line in a data file.
    s = line_in.split('$')
    try:
        return int(s[0]), s[1], int(s[2])
    except:
        # print('Debug: Tried to split \"{}\" but ended up with \"{}\".'.format(line_in, s[0]))
        return None, None, None


def merge_lines(orig, tran):
    # Merge two lines.
    o_index, o_body, part = get_index_and_body(orig)
    t_index, t_body, _ = get_index_and_body(tran)
    if o_index is None or t_index is None:
        return None
    if o_index != t_index:  # Check that index in both files are identical.
        # print('Warning: Failed to merge line {} (original) with line {} (translated).'.format(o_index, t_index))
        return None
    return '{}${}${}'.format(o_body, t_body, part)


def merge_files(training, full):
    # Merge two files and export the result.
    f_orig = io_service.get_filepath(training=training, full=full, original=True)
    f_tran = io_service.get_filepath(training=training, full=full, original=False)

    orig = io_service.get_lines_in_file(f_orig)
    tran = io_service.get_lines_in_file(f_tran)
    lines = []
    if len(orig) != len(tran):
        print('Warning: {} has {} lines while {} has {} lines.'.format(f_orig, len(orig), f_tran, len(tran)))
    for i in range(min(len(orig), len(tran))):
        s = merge_lines(orig[i], tran[i])
        if s is not None:
            lines.append(s)
        else:
            print('Error: Failed to merge line {} in {} and {}'.format(i, f_orig, f_tran))
    if len(lines) < 1:
        print('Error: No lines where merged.')
        return False

    # Export new merged file.
    f_merg = io_service.get_filepath(training=training, full=full, merged=True)
    io_service.export_lines_to_file(f_merg, lines)
    print('Debug: Successfully exported {}.'.format(f_merg))
    return True


def merge_dir(training):
    # Merge files in either data/training or data/validation directory.
    merge_files(training, True)  # Merge full sentences.
    merge_files(training, False)  # Merge segmented/augmented data.


def merge_all_files():
    # Merge all data files (there are eight in total) resulting in four new merged files.
    print('data_generator.py -> merge_all_files()')
    merge_dir(True)  # Merge files in training directory.
    merge_dir(False)  # Merge files in validation directory.


def main():
    app_name = 'IKT441 Project - Data Merger'
    print(' --- {} --- '.format(app_name))

    merge_all_files()


if __name__ == '__main__':
    main()
