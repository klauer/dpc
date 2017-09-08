import collections
import numpy as np
from hxn_db_config import db
import logging


logger = logging.getLogger(__name__)


def _eval_scan_args(scan_args):
    '''Evaluate scan arguments, replacing OphydObjects with NamedObjects'''

    class NamedObject:
        def __init__(self, name):
            self.name = name

    def no_op():
        def no_op_inner(*args, name=None, **kwargs):
            if name is not None:
                return NamedObject(name)

        return no_op_inner

    return eval(scan_args, collections.defaultdict(no_op))


step_1d = ('InnerProductAbsScan', 'HxnInnerAbsScan',
           'InnerProductDeltaScan', 'HxnInnerDeltaScan',
           'AbsScan', 'HxnAbsScan',
           'DeltaScan', 'HxnDeltaScan')

step_2d = ('OuterProductAbsScan', 'HxnOuterAbsScan')
fly_scans = ('FlyPlan1D', 'FlyPlan2D')


def get_scan_info(header):
    start_doc = header['start']
    scan_args = start_doc['scan_args']
    scan_type = start_doc['scan_type']
    motors = None
    range_ = None
    scan_args = start_doc.get('scan_args', {})
    pyramid = False

    if scan_type in fly_scans:
        logger.debug('Scan %s (%s) is a fly scan (%s)', start_doc.scan_id,
                     start_doc.uid, scan_type)
        dimensions = start_doc['dimensions']
        motors = start_doc['axes']
        pyramid = start_doc['fly_type'] == 'pyramid'
        try:
            range_ = start_doc['scan_range']
        except KeyError:
            try:
                range_ = [(float(scan_args['scan_start']),
                           float(scan_args['scan_end']))]
            except (KeyError, ValueError):
                pass
    elif scan_type in step_2d:
        logger.debug('Scan %s (%s) is an ND scan (%s)', start_doc.scan_id,
                     start_doc.uid, scan_type)
        # 2D mesh scan
        scan_args = _eval_scan_args(scan_args['args'])
        motors = [arg.name for arg in scan_args[::5]]
        dimensions = scan_args[3::5]
        range0 = scan_args[1::5]
        range1 = scan_args[2::5]
        range_ = list(zip(range0, range1))
    elif scan_type in step_1d or 'num' in start_doc:
        logger.debug('Scan %s (%s) is a 1D scan (%s)', start_doc.scan_id,
                     start_doc.uid, scan_type)
        # 1D scans
        dimensions = [int(scan_args['num'])]
        try:
            motors = [_eval_scan_args(scan_args['motor']).name]
        except Exception:
            motors = []
    else:
        msg = 'Unrecognized scan type (uid={} {})'.format(start_doc.uid,
                                                          scan_type)
        raise RuntimeError(msg)

    num = np.product(dimensions)

    return {'num': num,
            'dimensions': dimensions,
            'motors': motors,
            'range': range_,
            'scan_args': scan_args,
            'pyramid': pyramid,
            }


class Scan(object):
    def __init__(self, header):
        self.header = header
        self.start_doc = header['start']
        self.descriptors = header['descriptors']
        self.key = None
        for key, value in get_scan_info(self.header).items():
            logger.debug('Scan info %s=%s', key, value)
            setattr(self, key, value)

    @property
    def filestore_keys(self):
        for desc in self.descriptors:
            for key, info in desc['data_keys'].items():
                try:
                    external = info['external']
                except KeyError:
                    continue

                try:
                    source, info = external.split(':', 1)
                except Exception:
                    pass
                else:
                    source = source.lower()
                    if source in ('filestore', ):
                        yield key

    @property
    def scan_id(self):
        return self.start_doc['scan_id']

    def __repr__(self):
        return '{}(scan_id={})'.format(self.__class__.__name__, self.scan_id)

    def __iter__(self):
        if self.key:
            for event in self.header.db.fetch_events(self.header, fill=False):
                yield event['data'][self.key]
