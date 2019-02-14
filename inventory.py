import numpy as np
from collections import Counter
from tabulate import tabulate


class Warehouse():
    """
    Warehouse: class to organize inventory and inventory access methods
    """
    def __init__(self, dimx=16, dimy=5, dimz=2):
        self.DIMX = dimx
        self.DIMY = dimy
        self.DIMZ = dimz
        self.NUMBINS = dimx * dimy * dimz
        self.inventory = Counter()
        self.bins = [['-' for k in range(dimx * dimy)]
                     for z in range(dimz)]

    def xyz_to_index(self, x, y, z):
        """ return the bin index for coord (x,y,z) """
        return np.ravel_multi_index([x, y, z],
                                    [self.DIMX, self.DIMY, self.DIMZ])

    def index_to_xyz(self, idx):
        """ return the xyz coord for the linear index idx"""
        return np.unravel_index(idx,
                                [self.DIMX, self.DIMY, self.DIMZ])

    def place(self, c):
        """
        Add character c to the inventory and find the first
        available location. Return the x,y,z location as a tuple

        Return None if unsuccessful
        """
        try:
            index = self.bins.index('-')
        except ValueError:
            return None         # no room left

        self.bins.insert(index, c)
        self.inventory.update(c)
        return self.index_to_xyz(index)

    def pick(self, c):
        """
        Find and remove the first available character c. Return
        x,y,z location as a tuple.

        Return None if unsuccessful
        """
        if self.inventory[c] == 0:
            return None

        try:
            index = self.bins.index(c)
        except ValueError:
            return None

        self.bins[index] = '-'
        self.inventory.subtract(c)
        return self.index_to_xyz(index)

    def count(self, c):
        """ Return the number of c's in inventory"""
        return self.inventory[c]

    def print_inventory(self):
        """ Print the inventory in tabular form"""
        print(self.__str__())

        header = [x for x in range(self.DIMX)]
        rowids = [y for y in range(self.DIMY)]
        for z in range(self.DIMZ):
            print("\nZ = {}".format(z))
            rows = [[self.bins[self.xyz_to_index(x, y, z)] for x in range(self.DIMX)] for y in range(self.DIMY)]
            print(tabulate(rows, headers=header,
                           showindex=rowids, tablefmt="fancy_grid"))

    def __str__(self):
        header = ['part', 'count']
        rows = [[k, v] for k, v in self.inventory.items()]
        return tabulate(rows, headers=header)

    def __repr__(self):
        return self.__str__()


# TEST
# w = Warehouse()
# for c in 'abcasdflkajsdfoiahsdfoiauhwerljakhsdflkjahetoihasdf':
#     w.place(c)
