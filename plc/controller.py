from inventory import Inventory
from pycomm.ab_comm.slc import Driver
from time import sleep
from functools import wraps

"""
Project Params
Start:        B3:7/0  (1)
Pick/Place:   B3:7/1  Pick = 0, Place = 1 (2)
ZLoc:         B3:7/2  (4)
Done:         B3:7/3  (8)
XLoc:         N7:10
YLoc:         N7:11
"""

DRY_RUN = 'dry-run'
PRODUCTION = 'production'
BITFIELD = 'B3:7'
XLOC = 'N7:10'
YLOC = 'N7:11'
ZLOC = 'B3:7/2'

START = 'B3:7/0'
PICK_PLACE = 'B3:7/1'
DONE = 'B3:7/3'
PICK = 1
PLACE = 0
DIMX = 16
DIMY = 5
DIMZ = 2

class ConnectionException(Exception):
    """
    Exception raised when communication connection fails.

        Attributes:
            expr -- expression in which error occured
            msg  -- error explanation
    """
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg


class Simulator(object):

    def __init__(self):
        self.tags = {DONE: 1}
        self.tag_lookup = {DONE: 'DONE',
                           START: 'START',
                           PICK_PLACE: 'PICK_PLACE',
                           XLOC: 'XLOC',
                           YLOC: 'YLOC',
                           ZLOC: 'ZLOC',
                           }

    def tag_string(self):
        """
        Print a user-friendly version of tag values
        """
        tag_string = '{'
        for key, val in self.tags.items():
            tag_string += self.tag_lookup[key] + ':' + str(val) + ', '
        tag_string += '}'
        return tag_string

    def open(self, ip_address):
        return True

    def write_tag(self, tag, val):
        self.tags.update({tag: val})
        print('write_tag: ' + self.tag_lookup[tag] + ': ' + str(val))
        print('|')
        print('---->' + self.tag_string())

    def read_tag(self, tag):
        try:
            val = self.tags[tag]
        except KeyError:
            val = None

        print('read_tag:' + self.tag_lookup[tag])
        print('|')
        print('---->' + self.tag_string())

        return val


class Controller(Inventory):

    def __init__(self, ip_address=None, mode=DRY_RUN):
        self.ip_address = ip_address or None
        self.mode = mode
        self.done = False

        if self.mode == PRODUCTION:
            self.plc = Driver()
        else:
            self.plc = Simulator()
        try:
            connected = self.plc.open(self.ip_address)
            if not connected:
                msg = 'Failed to connect to {}'.format(self.ip_address)
                raise ConnectionException(msg)
        except ConnectionException as e:
            print(e.msg)

        self.write_tag = self.plc.write_tag
        self.read_tag = self.plc.read_tag

        super(Controller, self).__init__(dimx=DIMX,
                                         dimy=DIMY,
                                         dimz=DIMZ)

    def write_multi(self, in_list):
        for tag, value in in_list:
            self.write_tag(tag, value)

    def wait_till_done(self):
        """
        Loop until Done bit is set
        """
        while True:
            if self.read_tag(DONE):
                break
            sleep(0.1)

    def place(self, c):
        """
        Call super method to get empty location and then send params to PLC
        """
        self.wait_till_done()
        try:
            x, y, z = super(Controller, self).place(c)
            self.write_multi([
                (XLOC, x),
                (YLOC, y),
                (ZLOC, z),
                (PICK_PLACE, PLACE),
                (START, 1)])
        except TypeError:
            print('Inventory full!!!')

        print(self.inventory)

    def pick(self, c):
        """
        Call super method to get pick location and send params to PLC
        """
        self.wait_till_done()
        try:
            x, y, z = super(Controller, self).pick(c)
            self.write_multi([
                (XLOC, x),
                (YLOC, y),
                (ZLOC, z),
                (PICK_PLACE, PICK),
                (START, 1)])
        except TypeError:
            msg = 'Attempted to pick item '
            msg += '"{}", but none are available'.format(c)
            print(msg)
        print(self.inventory)
