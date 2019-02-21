import unittest

from mhlib.solution import BoolVectorSolution, one_point_crossover, multi_point_crossover


class TestSolution(BoolVectorSolution):
    """Solution to a MAXSAT instance.

    Attributes
        - inst: associated MAXSATInstance
        - x: binary incidence vector
    """

    def __init__(self, n: int):
        super().__init__(n)

    def copy(self):
        sol = TestSolution(len(self.x))
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        return 0

    def check(self):
        """Check if valid solution.

        :raises ValueError: if problem detected.
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


class OnePointCrossoverTestCase(unittest.TestCase):
    def test_ok(self):
        a = TestSolution(5)
        b = TestSolution(5)

        a.x.fill(True)
        b.x.fill(False)

        # Expected values
        ax = [True, True, False, False, False]
        bx = [False, False, True, True, True]

        one_point_crossover(a, b, 2)

        # Comparison
        for i in range(0, 5):
            self.assertEqual(a.x[i], ax[i])
            self.assertEqual(b.x[i], bx[i])

    def test_pos_to_small(self):
        a = TestSolution(2)
        b = TestSolution(2)
        self.assertRaises(AssertionError, lambda: one_point_crossover(a, b, 0))

    def test_pos_to_large(self):
        a = TestSolution(2)
        b = TestSolution(2)
        self.assertRaises(AssertionError, lambda: one_point_crossover(a, b, 2))

class MultiPointCrossoverTestCase(unittest.TestCase):
    def test_single_ok(self):
        a = TestSolution(5)
        b = TestSolution(5)

        a.x.fill(True)
        b.x.fill(False)

        # Expected values
        ax = [True, True, False, False, False]
        bx = [False, False, True, True, True]

        multi_point_crossover(a, b, [2])

        # Comparison
        for i in range(0, 5):
            self.assertEqual(a.x[i], ax[i])
            self.assertEqual(b.x[i], bx[i])

    def test_multi_ok(self):
        a = TestSolution(5)
        b = TestSolution(5)

        a.x.fill(True)
        b.x.fill(False)

        # Expected values
        ax = [True, True, False, False, True]
        bx = [False, False, True, True, False]

        multi_point_crossover(a, b, [2,4])

        # Comparison
        for i in range(0, 5):
            self.assertEqual(a.x[i], ax[i])
            self.assertEqual(b.x[i], bx[i])

if __name__ == '__main__':
    unittest.main()
