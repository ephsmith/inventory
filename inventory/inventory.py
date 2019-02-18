import numpy as np
from collections import Counter
from tabulate import tabulate


class Inventory(object):
    """
    Inventory: class to organize inventory and inventory access methods
    """
    def __init__(self, dimx=16, dimy=5, dimz=2):
        self.DIMX = dimx
        self.DIMY = dimy
        self.DIMZ = dimz
        self.NUMBINS = dimx * dimy * dimz
        self.inventory = Counter()
        self.bins = [['-' for k in range(dimx * dimy)]
                     for z in range(dimz)]

    def xy_to_index(self, x, y):
        """ return the bin index for coord (x,y) """
        return np.ravel_multi_index([x, y],
                                    [self.DIMX, self.DIMY])

    def index_to_xy(self, idx):
        """ return the x,y coord for the linear index idx"""
        return np.unravel_index(idx,
                                [self.DIMX, self.DIMY])

    def find_empty(self, c):
        """
        find the first empty bin for character c with the priority
        that like like items be stored at a higher z-depth.

        Fill priority is low to high
        """
        indices = []
        zloc = None
        index = None
        # is c in the inventory?
        if self.inventory[c] > 0:

            # find c at z depth 0 first
            for z in range(self.DIMZ-1):
                indices = [k for k, x in enumerate(self.bins[z])
                           if x == c and self.bins[z+1][k] == '-']
                if len(indices) > 0:
                    index = indices[0]
                    zloc = z + 1
                    return zloc, index
        if zloc is None and index is None:
            try:
                index = self.bins[0].index('-')
                zloc = 0
            except ValueError:
                index = None
                zloc = None
        return zloc, index

    def find_top(self, c):
        """
        find the topmost character c
        """
        zloc = None
        index = None
        # is c in the inventory?
        if self.inventory[c] > 0:

            # find c at z depth 0 first
            for z in reversed(range(self.DIMZ)):
                indices = [k for k, x in enumerate(self.bins[z])
                           if x == c]
                if len(indices) > 0:
                    index = indices[0]
                    zloc = z
                    break
        return zloc, index

    def place(self, c):
        """
        Add character c to the inventory and find the first
        available location. Return the x,y,z location as a tuple

        Return None if unsuccessful
        """
        z, index = self.find_empty(c)

        if z is not None and index is not None:
            self.bins[z].insert(index, c)
            self.inventory.update(c)
            xy = self.index_to_xy(index)
            return (xy[0], xy[1], z)

    def pick(self, c):
        """
        Find and remove the first available character c. Return
        x,y,z location as a tuple.

        Return None if unsuccessful
        """
        if self.inventory[c] == 0:
            return None

        z, index = self.find_top(c)

        self.bins[z][index] = '-'
        self.inventory.subtract(c)
        xy = self.index_to_xy(index)
        return (xy[0], xy[1], z)

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
            rows = [[self.bins[z][self.xy_to_index(x, y)]
                     for x in range(self.DIMX)] for y in range(self.DIMY)]
            print(tabulate(rows, headers=header,
                           showindex=rowids, tablefmt="fancy_grid"))

    def __str__(self):
        header = ['part', 'count']
        rows = [[k, v] for k, v in self.inventory.items()]
        return tabulate(rows, headers=header)

    def __repr__(self):
        return self.__str__()
