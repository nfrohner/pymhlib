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

    def local_improve(self, par, result):
        """Perform k_flip_local_search."""
        del result
        self.k_flip_local_search(par, False)

    def shaking(self, par, result):
        """Scheduler method that performs shaking by flipping par random positions."""
        del result
        for i in range(par):
            p = random.randrange(0, self.inst.n)
            self.x[p] = not self.x[p]
        self.invalidate()

    def k_flip_local_search(self, k: int, best_improvement) -> bool:
        """Perform one major iteration of a k-flip local search.

        If best_improvement is set, the neighborhood is completely searched and a best neighbor is kept;
        otherwise the search terminates in a first-improvement manner, i.e., keeping a first encountered
        better solution.

        Returns True if an improved solution has been found.
        """
        x = self.x
        assert 0 < k <= len(x)
        better_found = False
        best_sol = self.copy()
        p = np.full(k, -1)  # flipped positions
        # initialize
        i = 0  # current index in p to consider
        while i >= 0:
            # evaluate solution
            if i == k:
                self.invalidate()
                if self.is_better(best_sol):
                    if not best_improvement:
                        return True
                    best_sol.copy_from(self)
                    better_found = True
                i -= 1  # backtrack
            else:
                if p[i] == -1:
                    # this index has not yet been placed
                    p[i] = (p[i-1] if i > 0 else -1) + 1
                    x[p[i]] = not x[p[i]]
                    i += 1  # continue with next position (if any)
                elif p[i] < len(x) - (k - i):
                    # further positions to explore with this index
                    x[p[i]] = not x[p[i]]
                    p[i] += 1
                    x[p[i]] = x[p[i]]
                    i += 1
                else:
                    # we are at the last position with the i-th index, backtrack
                    x[p[i]] = not x[p[i]]
                    p[i] = -1  # unset position
                    i -= 1
        if better_found:
            self.copy_from(best_sol)
            self.invalidate()
            return better_found


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

