from time import sleep
import plc
from plc import Controller

w = Controller(mode=plc.DRY_RUN)

for c in 'aabbccddeeffgg':
    w.place(c)

w.print_inventory()

for c in 'abcdefg':
    w.pick(c)

w.print_inventory()

# c = SlcDriver()
# if c.open('192.168.200.20'):
#     print('Connected')
#     b3_backup = c.read_tag('B3:0')
#     print('B3:0 = ', b3_backup)
#     print('Writing B3:0 = 256, F8:0 = 2400')
#     c.write_tag('F8:0', 100)
#     c.write_tag('B3:0', 256)
#     print('Done')
#     sleep(10)
#     c.write_tag('B3:0', b3_backup)
#     c.write_tag('F8:0', -2400)
