from inventory import Inventory
from pycomm.ab_comm.slc import Driver
from time import sleep
from threading import Thread

"""
Project Params
Start:        B3:7/0  (1)
Pick/Place:   B3:7/1  Pick = 0, Place = 1 (2)
ZLoc:         B3:7/2  (4)
Done:         B3:7/3  (8)
XLoc:         N7:10
YLoc:         N7:11
"""

BITFIELD = 'B3:7'
DONE = 0x08
XLOC = 'N7:10'
YLOC = 'N7:11'


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
        self.tags = {BITFIELD: 8}

    def open():
        return True

    def write_tag(self, d):
        self.tags.update(d)

    def read_tag(self, s):
        try:
            val = self.tags[s]
        except KeyError:
            val = None
        return val


class Controller(Inventory):

    def __init__(self, ip_address=None, mode='dry-run'):
        self.ip_address = ip_address or None
        self.mode = mode
        self.done = False

        if self.mode is not 'dry-run':
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

        self.done_thread = Thread(target=self.check_done)
        self.done_thread.start()

    def check_done(self):
        """
        check_done is blocking and should be run in a separate thread
        """
        while True:
            if self.plc.read_tag(BITFIELD) & DONE:
                self.done = True
            else:
                self.done = False
                sleep(0.5)

    def place(c):
        """
        Call super method to get location and then send params to PLC
        """
        x, y, z = position = super(Controller, self).place(c)
