"""
SPEC file format writer
... and partial reader, but it's recommended to use Specfile instead
"""
import re
import os
import time
import itertools
import numpy as np

TIME_FORMAT = '%a %b %d %H:%M:%S %Y'


def split_sequence(iterable, size):
    """
    Take an iterable sequence (not just lists), and split it into
    equal sized chunks (<= size).
    """
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


class SPECFileError(Exception):
    pass


class SPECFileMotorListError(SPECFileError):
    pass


class SPECFileWriter(object):
    """
    Writes SPEC-format files for a scan.
    Simple example:

        motors = ['m0', 'm1']
        data_names = ['test', 'one', 'two']
        a = SPECFileWriter('test.txt', comment='my comment', motors=motors)
        a.write_scan_start(command='dscan something', seconds=1)
        a.write_motor_positions([3, 4])
        a.write_scan_data_start(data_names)
        for d1, d2, d3 in zip(range(10), range(2,12), range(10,20)):
            a.write_scan_data([d1,d2,d3])
        a.finish_scan()

    Remember that SPEC scan counts are 0-based -- this means that 0 data points
    is actually 1 data point in EPICS.
    """

    COLUMNS = 8

    def __init__(self, filename, starting_number=None, comment='', motors=[]):
        if not filename:
            raise SPECFileError('Must specify filename')

        self.filename = os.path.abspath(filename)
        self._motors = list(sorted(motors))
        self._comment = comment

        self._data_lines = 0  # how many lines in the current scan there are
        # Scan headers are not written until there's at least one data point
        # available. (otherwise, the C-based spec file reader can fail)
        self._buffer_write = False
        self._buffer = []

        if os.path.exists(filename) and os.path.getsize(filename) > 1:
            if not check_motor_list(filename, motors):
                raise SPECFileMotorListError('Motor list does not match')

            with SPECFileReader(filename) as reader:
                self.start_time = reader.epoch()

                if starting_number is None:
                    self.scan_number = get_last_scan_number(reader)
                else:
                    self.scan_number = starting_number

            print('SPECFileWriter: Appending to "%s"' % filename)
            self._f = open(filename, 'at')
            self._fix_ending_newlines(filename)

            self._header_written = True
        else:
            print('SPECFileWriter: New SPEC-format file "%s"' % filename)
            if starting_number is None:
                starting_number = 0

            self.scan_number = starting_number
            self._f = open(filename, 'wt')
            self._header_written = False

    def _get_motors(self):
        return list(self._motors)

    def _set_motors(self, motors):
        if self._header_written:
            if motors != self._motors:
                raise SPECFileMotorListError(
                    'Cannot change motor list when header is already written')
            return

        self._motors = list(motors)

    def _fix_ending_newlines(self, filename=None, amount=2):
        if filename is None:
            filename = self.filename

        count = self._check_end_lines(filename, amount)
        for i in range(amount - count):
            self.blank_line()

    def _check_end_lines(self, filename, ending=2, char='\n'):
        try:
            f = open(filename, 'rt')
            # Seek two bytes before the end of the file
            f.seek(-ending, 2)
            return f.read().count(char)
        except:
            return 0

    def close(self):
        if self._f:
            self._f.close()
            self._f = None

    def write_line(self, line):
        line = '%s\n' % line
        if self._buffer_write:
            self._buffer.append(line)
        else:
            if self._buffer:
                self._f.writelines(self._buffer)
                self._buffer = []

            self._f.write(line)

    def write_info(self, tag, data):
        self.write_line('#%s %s' % (tag, data))

    def write_scan_start(self, number=None, command='', scan_info=None, seconds=None):
        """
        Writes the following:
        Scan information, date, wait time

        In the format:
        #S 1  hklscan  0.9 1.1  0 0  0 0  20 1
        #D Wed Feb 17 19:25:55 1994
        #T 1  (Seconds)
        """
        if not self._header_written:
            self._write_header(self._comment, self._motors)

        if self._data_lines > 0:
            self.finish_scan()

        if number is None:
            self.scan_number += 1
            number = self.scan_number

        # Buffer this scan header until data comes in
        self._buffer_write = True

        # S - scan number and command
        self.write_info('S', '%d  %s' % (number, command))
        self.write_date()
        if seconds is not None:
            # T - scan/settling time
            self.write_info('T', '%f (Seconds)' % (seconds, ))

    def write_scan_data_start(self, columns):
        if columns is None:
            columns = []

        self._data_lines = 0

        # N - Number of columns
        self.write_info('N', len(columns))
        # L - Column names (double-space separated)
        self.write_info('L', '  '.join(columns))

    def write_motor_positions(self, positions):
        if not positions:
            return

        for i, pos in enumerate(split_sequence(positions, self.COLUMNS)):
            # P{num} - starting motor positions (that aren't necessarily being
            # scanned)
            tag = 'P%d' % i
            self.write_info(tag, ' '.join([str(p) for p in pos]))

    def write_scan_data(self, data):
        self._buffer_write = False
        self._data_lines += 1
        self.write_line(' '.join([str(d) for d in data]))

    def write_mca_calib(self, a, b, c):
        """
        Ref: http://www.esrf.eu/blissdb/macros/macdoc.py?macname=saveload.mac

        CALIBRATION = a + b*CHANNEL + c*CHANNEL^2
        """
        self.write_line('#@CALIB  %.7g %.7g %.7g' % (a, b, c))

    def write_mca_data(self, data, calibration=None, first=False):
        """
        """
        if data is None or np.size(data) == 0:
            return

        self._buffer_write = False
        self._data_lines += 1
        # does it really support multiple sets of MCA data per scan?
        #if isinstance(data, (list, tuple)):
        #    if isinstance(data[0], np.ndarray):
        #        for array_ in data:
        #            self.write_mca_data(array_, calibration=calibration, first=first)
        #        return

        if first:
            # MCA format (full line)
            self.write_line('#@MCA %%%dC' % len(data))

            # number of channels, first idx, last idx, reduction coefficient
            self.write_line('#@CHANN  %d 0 %d 1' % (len(data), len(data) - 1))

            if calibration:
                self.write_mca_calib(*calibration)

        self.write_line('@A %s' % ' '.join([str(d) for d in data]))

    @property
    def date_string(self):
        # return time.strftime('%c')
        # TODO maybe this ^^ is more appropriate for other locales?
        return time.strftime(TIME_FORMAT)

    def _write_starting_seconds(self):
        self.start_time = time.time()
        self.write_info('E', int(self.start_time))

    def write_date(self):
        self.write_info('D', self.date_string)

    def write_timestamp(self):
        self.write_info('E', self.date_string)

    def write_scan_aborted(self, points_written=None):
        self._buffer_write = False

        if points_written is not None:
            self._data_lines = points_written

        if self._data_lines > 0:
            self.write_info('C', '%s.  Scan aborted after %d points.' %
                           (self.date_string, self._data_lines))
            self.blank_line()
            self._data_lines = 0
            self._f.flush()
        else:
            # Cancelled before it even started. Don't even write the
            # buffered header.
            self._buffer = []

    def blank_line(self):
        self.write_line('')

    def _write_header(self, comment='', motors=[]):
        self._header_written = True
        self.write_info('F', self.filename)
        self._write_starting_seconds()
        self.write_date()
        if comment is not None:
            for line in comment.split('\n'):
                self.write_info('C', line)  # TODO multiple lines valid?

        for i, mot in enumerate(split_sequence(motors, self.COLUMNS)):
            # O{num} - motor names (that aren't necessarily being scanned)
            tag = 'O%d' % i
            self.write_info(tag, '  '.join([str(m) for m in mot]))

        self.blank_line()
        self._data_lines = 0

    def finish_scan(self):
        self._fix_ending_newlines()

        self._data_lines = 0
        self._f.flush()


class SPECFileReader(object):
    """
    NOTE: Really untested except for reading the header.
    (just needed a safe, quick way to check the motor list.)
    """

    def __init__(self, filename, parse_data=True):
        if not os.path.exists(filename):
            raise ValueError('Invalid SPEC filename')

        self._f = open(filename, 'rt')
        self._in_scan = False
        self._buffer_lines = []
        self._buffer_groups = []
        self._scans = []
        self._scan = None
        self._eof = False
        self._epoch = time.time()
        self._mca_line = False
        self._parse_data = parse_data

        self.spec_filename = filename
        self.motors = []
        self.comment = ''

        self._read_header()

    def close(self):
        if self._f:
            self._f.close()
            self._f = None

    def scanno(self):
        return 0  # TODO

    def epoch(self):
        return self._epoch

    def _read_line(self):
        while True:
            while self._buffer_lines:
                line = self._buffer_lines.pop(0)
                yield line

            line = self._f.readline()
            if line == '':
                yield '#S'  # TODO: fix so this isn't necessary
                self._eof = True
                break

            line = line.strip()
            if line and (line.startswith('#') or self._in_scan):
                yield line

    def _read_group(self):
        while self._buffer_groups:
            yield self._buffer_groups.pop(0)

        current_tag = None
        group = []
        tag_done = False

        for line in self._read_line():
            if self._eof:
                if current_tag is not None:
                    yield current_tag.upper(), group

                break

            if line.startswith('#'):
                # print('-> %s' % line)
                if ' ' in line[1:]:
                    tag, info = line[1:].split(' ', 1)
                else:
                    tag, info = line[1], ''

                info = info.lstrip()

                m = re.match('([@a-zA-Z]+)(\d*)', tag)
                if m:
                    tag, tag_index = m.groups()
                else:
                    tag_index = None

                # print('current', current_tag, 'read tag', tag, 'index',
                # tag_index)
                if current_tag is None:
                    current_tag = tag
                    group.append(info)
                    if tag_index is None:
                        tag_done = True
                else:
                    if tag == current_tag:
                        group.append(info)
                    else:
                        self._buffer_lines.append(line)
                        tag_done = True

            elif self._in_scan:
                if self._parse_data:
                    self._parse_scan_line(line)
                else:
                    self._scan['unparsed'].append(line)

            if tag_done:
                yield current_tag.upper(), group

                current_tag = None
                group = []
                tag_done = False

    def _parse_list(self, list_):
        list_ = re.sub('\s+', ',', list_)
        return list_.split(',')

    def _parse_header_F(self, spec_filename):
        self.spec_filename = spec_filename

    def _parse_header_D(self, date_):
        self._epoch = time.mktime(time.strptime(date_, TIME_FORMAT))

    def _parse_header_list_O(self, motors):
        self.motors = motors

    def _parse_header_C(self, comment):
        self.comment = comment

    def _read_section(self, section, end_tags=None, ignore_first_tag=False):
        first_tag = True
        for tag, lines in self._read_group():
            if self._eof:
                return

            if end_tags is not None and tag in end_tags or self._eof:
                if not (first_tag and ignore_first_tag):
                    self._buffer_groups.insert(0, (tag, lines))
                    break

            # print('section', section, tag, lines)
            lines = '  '.join(lines)
            fcn_name = '_parse_%s_list_%s' % (section, tag)
            if hasattr(self, fcn_name):
                fcn = getattr(self, fcn_name)
                fcn(self._parse_list(lines))

            fcn_name = '_parse_%s_%s' % (section, tag)
            if hasattr(self, fcn_name):
                # print('calling', fcn_name)
                fcn = getattr(self, fcn_name)
                fcn(lines)

            first_tag = False

    _read_header = lambda self: self._read_section('header', end_tags=['S'])

    def _parse_scan_S(self, scan_info):
        number, command = scan_info.split(' ', 1)

        self._in_scan = True
        self._scan = {
            'lines': [],
            'mca_data': [],
            'columns': [],
            'time': 0,
            'hkl': [0, 0, 0],
            'fourc': [],
            'positions': [],
        }

        if not self._parse_data:
            self._scan['unparsed'] = []

        self._scan['number'] = number
        self._scan['command'] = command.strip()

        self._scans.append(self._scan)

    def _parse_scan_list_L(self, columns):
        self._scan['columns'] = columns

    def _parse_scan_D(self, date_):
        self._scan['time'] = time.mktime(time.strptime(date_, TIME_FORMAT))

    def _parse_scan_list_G(self, fourc_info):
        self._scan['fourc'] = fourc_info

    def _parse_scan_list_Q(self, hkl):
        self._scan['hkl'] = hkl

    def _parse_scan_line(self, line):
        if line.startswith('@A'):
            line = line.split(' ', 1)[1]

            # Line is now everything after @A
            data = [float(f) for f in line.split(' ')]

            # If the MCA data is spread on multiple lines, extend the previous
            # data, otherwise it's a new set
            if self._mca_line:
                self._scan['mca_data'][-1].extend(data)
            else:
                self._scan['mca_data'].append(data)

            self._mca_line = line.endswith('\\')
        else:
            try:
                line = line.replace('None', '0.0')  # TODO during saving
                self._scan['lines'].append([float(f) for f in line.split(' ')])
            except Exception as ex:
                print('Bad scan line: %s' % line)

    def parse_data(self, scan):
        '''
        Parse the scan data after the file is loaded, if parse_data was set
        '''
        if self._parse_data:
            return

        if 'unparsed' not in scan:
            return

        self._scan = scan
        for line in scan['unparsed']:
            self._parse_scan_line(line)

        del scan['unparsed']

    def _parse_scan_list_P(self, positions):
        positions = [float(p) if p != 'None' else 0.0
                     for p in positions]  # TODO fix
        self._scan['positions'] = dict(zip(self.motors, positions))

    @property
    def scans(self):
        while True:
            scan = self.read_scan()
            if scan is None:
                break

            yield scan

    def read_scan(self):
        if self._eof:
            return None

        self._read_section('scan', end_tags=['S', ], ignore_first_tag=True)
        # should 'C' really be an end_tag, since you can use scan_on?
        # what does scan_on output look like?
        self._in_scan = False

        return self._scan

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def check_motor_list(filename, motors):
    try:
        sf = SPECFileReader(filename)
    except:
        return False
    return (motors == sf.motors)


def get_last_scan_number(filename):
    numbers = [0]
    try:
        if isinstance(filename, SPECFileReader):
            reader = filename
        else:
            reader = SPECFileReader(filename)

        for scan in reader.scans:
            try:
                numbers.append(int(scan['number']))
            except:
                pass

        return max(numbers)

    except Exception as ex:
        print('Failed to get last scan number: (%s) %s' % (filename, ex))
        return 0


if __name__ == '__main__':
    motors = ['m0', 'm1']
    data_names = ['test', 'one', 'two']
    a = SPECFileWriter('test.txt', comment='my comment', motors=motors)
    a.write_scan_start(command='dscan something', seconds=1)
    a.write_motor_positions([3, 4])
    a.write_scan_data_start(data_names)
    for d1, d2, d3 in zip(range(10), range(2, 12), range(10, 20)):
        a.write_scan_data([d1, d2, d3])
    a.finish_scan()

    # reader = SPECFileReader('/epics/data/aug_21_11')
    reader = SPECFileReader('../test_output')
    for scan in reader.scans:
        print(scan['number'], scan['command'], scan.keys(), scan['columns'])

    print(reader._scans[-1], len(reader._scans))
    print(reader._buffer_groups)
    print(reader._scans[-1]['lines'])
    print(reader._scans[-1]['columns'])
    print(reader._scan['columns'])
