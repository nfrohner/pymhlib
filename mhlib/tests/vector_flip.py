import unittest

import numpy as np

from mhlib.solution import BoolVectorSolution


class TestSolution(BoolVectorSolution):
    """Solution for testing the k-flip operation.

    Attributes
        - x: binary incidence vector
    """

    def copy(self):
        sol = TestSolution(len(self.x))
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        return np.sum(self.x)

    def construct(self, par, result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        del result
        self.initialize(par)


class KFlip (unittest.TestCase):

    def test_kflip_to_small(self):
        sol = TestSolution(5)
        self.assertRaises(AssertionError, lambda: sol.kflip(0))

    def test_kflip_to_big(self):
        sol = TestSolution(5)
        self.assertRaises(AssertionError, lambda: sol.kflip(6))

    def test_kflip_ok(self):
        sol = TestSolution(5)

        for _ in range(10):
            tmp = sol.copy()
            tmp.kflip(3)
            self.assertEqual(tmp.obj(), 3)

    def test_kflipall_ok(self):
        sol = TestSolution(5)
        allsolutions = sol.kflip_all(3)

        self.assertEqual(len(allsolutions), 10)

        for solution in allsolutions:
            self.assertEqual(solution.obj(), 3)


if __name__ == '__main__':
    unittest.main()
