import numpy as np


class Empty_Node(object):

    def __init__(self):
        self.left = None
        self.right = None
        self.size = -1
        self.p = None
        self.normal, self.inter = None, None

    def path_len(self, pt, e=0):
        if self.size != -1:
            return e + self.avg_external_len(self.size)
        if (pt-self.inter).dot(self.normal) < 0:
            return self.left.path_len(pt, e + 1)
        else:
            return self.right.path_len(pt, e + 1)

    def avg_external_len(self, size):
        if (self.size == 1) | (self.size == 0) | (size <= 1):
            return 0
        return 2 * np.log(size - 1) + 0.5772156649 - (2 * (size - 1) / size)

    def print_splits(self):
        if self.inter != None:
            print(self.normal, self.inter)
            self.left.print_splits()
            self.right.print_splits()


class Node(Empty_Node):

    def __init__(self, data, bound, e, l, dim, exlevel=0):
        super(Node, self).__init__()
        self.train(data, bound, e, l, dim)

    def train(self, data, bound, e, l, dim, exlevel=0):
        if (e >= l) | (len(data) <= 1):
            self.size = len(data)
        else:
            idxs = np.random.choice(range(dim), dim - exlevel - 1, replace=False)  # Picks the indices for which the normal vector elements should be set to zero acccording to the extension level.
            self.normal = np.random.normal(0, 1, dim)    # A random normal vector picked form a uniform n-sphere.
            self.normal[idxs] = 0
            self.inter = np.random.uniform(bound[:, 0], bound[:, 1])
            idx_l = (data-self.inter).dot(self.normal) < 0
            data_l = data[np.where(idx_l)[0]]
            data_r = data[np.where(1 - idx_l)[0]]
            self.p = len(data_l) / (len(data_l) + len(data_r))
            self.left = Node(data_l, bound, e + 1, l, dim)
            self.right = Node(data_r, bound, e + 1, l, dim)


class Copy_Node(Empty_Node):
    """docstring for Copy_Node"""

    def __init__(self, node):
        super(Copy_Node, self).__init__()
        self.size = node.size
        self.w = 0.5

        self.p = node.p
        self.normal = node.normal
        self.inter = node.inter
        if node.left is not None:
            self.left = Copy_Node(node.left)
        if node.right is not None:
            self.right = Copy_Node(node.right)

    def path_len(self, pt, e=0):
        if self.size != -1:
            return (e + self.avg_external_len(self.size)) * self.w
        if (pt-self.inter).dot(self.normal) < 0:
            return self.left.path_len(pt, e + 1)
        else:
            return self.right.path_len(pt, e + 1)

    def train(self, data):
        if self.inter is None:
            self.size = len(data)
        else:
            idx_l = (data-self.inter).dot(self.normal) < 0
            data_l = data[np.where(idx_l)[0]]
            data_r = data[np.where(1 - idx_l)[0]]
            self.left.train(data_l)
            self.right.train(data_r)

    def mutate(self, sigma, bound, dim, exlevel=0):
        if self.inter is not None:

            if np.random.uniform() < sigma * 4:
                idxs = np.random.choice(range(dim), dim - exlevel - 1, replace=False)  # Picks the indices for which the normal vector elements should be set to zero 
                self.normal = np.random.normal(0, 1, dim)  				# A random normal vector picked form a uniform n-sphere.
                self.normal[idxs] = 0
                self.inter = np.random.uniform(bound[:, 0], bound[:, 1])
            self.inter = self.inter + sigma * np.random.randn() * (bound[:, 1] - bound[:, 0])

            self.inter = np.amin([np.amax([self.inter, bound[:,0]], axis=0), bound[:,1]], axis=0)

            self.left.mutate(sigma, bound, dim)
            self.right.mutate(sigma, bound, dim)
        else:
            return

    def locate_crossover(self, p=0.3):
        rand = np.random.uniform()
        if self.inter is None:
            return None
        if rand < p / 2:
            return self.left
        elif rand < p:
            return self.right
        elif rand < p + (1 - p) / 2:
            return self.left.locate_crossover(p)
        else:
            return self.right.locate_crossover(p)

    def set_crossover(self, node, p=0.3):
        rand = np.random.uniform()
        if self.inter is None:
            return False
        if rand < p / 2:
            self.left = node
            return True
        elif rand < p:
            self.right = node
            return True
        elif rand < p + (1 - p) / 2:
            return self.left.set_crossover(node, p)
        else:
            return self.right.set_crossover(node, p)

