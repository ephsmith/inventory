from inventory import Inventory

w = Inventory()
for c in 'aaaaaabbbbbbccccccffffff':
    w.place(c)

w.print_inventory()

print("Picking 3 a's:")

for x in range(3):
    w.pick('a')

w.print_inventory()
