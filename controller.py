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

    def open():
        return True

    def write_tag(self, tag, val):
        self.tags.update({tag: val})

    def read_tag(self, tag):
        try:
            val = self.tags[tag]
        except KeyError:
            val = None
        return val


class Controller(Inventory):

    def __init__(self, ip_address=None, mode=DRY_RUN):
        self.ip_address = ip_address or None
        self.mode = mode
        self.done = False

        if self.mode == PRODUCTION:
            self.plc = Driver()
            try:
                connected = self.plc.open(self.ip_address)
                if not connected:
                    msg = 'Failed to connect to {}'.format(self.ip_address)
                    raise ConnectionException(msg)
            except ConnectionException as e:
                print(e.msg)
        else:
            self.plc = Simulator()

        self.write_tag = self.plc.write_tag
        self.read_tag = self.plc.read_tag

        self.done_thread = Thread(target=self.check_done)
        self.done_thread.start()
        super(Controller, self).__init__(dimx=16,
                                         dimy=5,
                                         dimz=2)

    def write_multi(self, in_list):
        for tag, value in in_list:
            self.write_tag(tag, value)

    def check_done(self):
        """
        check_done is blocking and should be run in a separate thread
        """
        try:
            while True:
                if self.plc.read_tag(DONE):
                    self.done = True
                else:
                    self.done = False
                    sleep(0.1)
        except KeyboardInterrupt:
            print('Done POLL method interrupted')
            return

    def wait_till_done(self):
        """
        Loop until Done bit is set
        """
        while not self.done:
            sleep(0.1)

    def place(self, c):
        """
        Call super method to get empty location and then send params to PLC
        """
        self.wait_till_done()
        x, y, z = super(Controller, self).place(c)
        self.write_multi([
            (XLOC, x),
            (YLOC, y),
            (ZLOC, z),
            (PICK_PLACE, PLACE),
            (START, 1)])

    def pick(self, c):
        """
        Call super method to get pick location and send params to PLC
        """
        self.wait_till_done()
        x, y, z = super(Controller, self).pick(c)
        self.write_multi((XLOC, x),
                         (YLOC, y),
                         (ZLOC, z),
                         (PICK_PLACE, PICK),
                         (START, 1))
