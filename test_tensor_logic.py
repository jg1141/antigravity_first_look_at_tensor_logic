import unittest
import numpy as np
from tensor_logic import Tensor, Equation, Program

class TestTensorLogic(unittest.TestCase):
    def test_tensor_creation(self):
        data = np.array([1, 2, 3])
        t = Tensor("T", ["i"], data)
        self.assertEqual(t.name, "T")
        self.assertEqual(t.indices, ["i"])
        np.testing.assert_array_equal(t.data, data)

    def test_equation_parsing(self):
        eq = Equation("C[i,k]", "A[i,j] * B[j,k]")
        self.assertEqual(eq.lhs_name, "C")
        self.assertEqual(eq.lhs_indices, ["i", "k"])
        self.assertEqual(len(eq.rhs_terms), 2)
        self.assertEqual(eq.rhs_terms[0], ("A", ["i", "j"]))
        self.assertEqual(eq.rhs_terms[1], ("B", ["j", "k"]))
        self.assertIsNone(eq.activation)

    def test_equation_parsing_activation(self):
        eq = Equation("Y[i]", "sigmoid(W[i,j] * X[j])")
        self.assertEqual(eq.activation, "sigmoid")
        self.assertEqual(len(eq.rhs_terms), 2)

    def test_execution_matmul(self):
        # C = A * B
        A = Tensor("A", ["i", "j"], np.array([[1, 2], [3, 4]]))
        B = Tensor("B", ["j", "k"], np.array([[5, 6], [7, 8]]))
        
        prog = Program()
        prog.add_tensor(A)
        prog.add_tensor(B)
        prog.add_equation("C[i,k] = A[i,j] * B[j,k]")
        prog.run()
        
        C = prog.tensors["C"]
        expected = np.dot(A.data, B.data)
        np.testing.assert_array_equal(C.data, expected)

    def test_execution_step(self):
        # Aunt example logic
        # Sister: (0,1)=1
        # Parent: (1,2)=1
        # Aunt: (0,2) should be 1
        sister = Tensor("Sister", ["x", "y"], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
        parent = Tensor("Parent", ["y", "z"], np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
        
        prog = Program()
        prog.add_tensor(sister)
        prog.add_tensor(parent)
        prog.add_equation("Aunt[x,z] = step(Sister[x,y] * Parent[y,z])")
        prog.run()
        
        aunt = prog.tensors["Aunt"]
        self.assertEqual(aunt.data[0, 2], 1)
        self.assertEqual(aunt.data[0, 1], 0)

if __name__ == '__main__':
    unittest.main()
