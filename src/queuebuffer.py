import numpy as np


class QueueBuffer:
    def __init__(self, size: int, initvector: np.ndarray):
        """
        :param size: size of a circular buffer
        :param initvector: vector with initial values for all buffer entries
        """
        self.size = size
        self.initvector = np.copy(initvector)
        self.buffer = np.zeros((size, *initvector.shape))
        self.buffer[:, ] = initvector
        self.idx = 0

    def reset(self):
        """
        Index to zero. Buffer values to initial
        """
        self.buffer[:, ] = self.initvector
        self.idx = 0

    def iterate(self, idx):
        """
        :param idx: index to iterate
        :return: index moved by 1
        """
        return (idx + 1) % self.size

    def add(self, element: np.ndarray):
        """
        :param element: data to push to the queue
        """
        if element.shape != self.initvector.shape:
            return
        self.buffer[self.idx] = np.copy(element) 
        self.idx = self.iterate(self.idx)

    def get_vector(self):
        """
        :return: buffer data as a flat vector
        """
        idx = self.idx
        vector = self.buffer[idx]
        for _ in range(self.size-1):
            idx = self.iterate(idx)
            vector = np.hstack((vector, self.buffer[idx]))
        return vector

    def get_sequential_input(self):
        """
        :return: consequential observations as one sequence
        """
        vector = self.get_vector()
        return vector.reshape((1, self.size, *self.initvector.shape))


if __name__ == '__main__':
    buf = QueueBuffer(3, -np.ones(3))
    print(buf.get_vector())
    buf.add(np.array([1, 0.5, 0.2]))
    buf.add(np.array([0.2, 0.3, -0.4]))
    print(buf.get_vector())

