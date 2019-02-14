from inventory import Warehouse

w = Warehouse()
for c in 'abcasdflkajsdfoiahsdfoiauhwerljakhsdflkjahetoihasdf':
    w.place(c)

w.print_inventory()
