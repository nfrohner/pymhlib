"""Demo application solving the Quadratic Assignment Problem (QAP)."""

import unittest

import numpy as np
import random

from mhlib.solution import BoolVectorSolution

class TestSolution(BoolVectorSolution):
    """Solution to a MAXSAT instance.

    Attributes
        - inst: associated MAXSATInstance
        - x: binary incidence vector
    """

    def copy(self):
        sol = TestSolution(len(self.x))
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        return np.sum(self.x)

    def check(self):
        """Check if valid solution.

        Raises ValueError if problem detected.
        """
        if len(self.x) != self.inst.n:
            raise ValueError("Invalid length of solution")
        super().check()

    def construct(self, par, result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        del result
        self.initialize(par)


def func(sol, sub):
    for pos in sub:
        sol.x[pos] = not sol.x[pos]

    sol.invalidate()
    return sol


class KFlip (unittest.TestCase):

    def test_kflip_to_small(self):
        sol = TestSolution(5)
        self.assertRaises(AssertionError, lambda:sol.kflip(0, func))

    def test_kflip_to_big(self):
        sol = TestSolution(5)
        self.assertRaises(AssertionError, lambda:sol.kflip(6, func))

    def test_kflip_ok(self):
        sol = TestSolution(5)

        for _ in range(10):
            tmp = sol.kflip(3, func)
            self.assertEqual(tmp.obj(), 3)

    def test_kflipall_ok(self):
        sol = TestSolution(5)
        all = sol.kflip_all(3, func)

        self.assertEqual(len(all), 10)

        for tmp in all:
            self.assertEqual(tmp.obj(), 3)


if __name__ == '__main__':
    unittest.main()

