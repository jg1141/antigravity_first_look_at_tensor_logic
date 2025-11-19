import numpy as np
import re

class Tensor:
    def __init__(self, name, indices, data):
        """
        Initialize a Tensor.
        :param name: Name of the tensor (str)
        :param indices: List of index names (list of str)
        :param data: Numpy array containing the data
        """
        self.name = name
        self.indices = indices
        self.data = np.array(data)
        if len(indices) != self.data.ndim:
            raise ValueError(f"Number of indices ({len(indices)}) must match data rank ({self.data.ndim})")

    def __repr__(self):
        return f"Tensor({self.name}, {self.indices}, shape={self.data.shape})"

class Equation:
    def __init__(self, lhs_str, rhs_str):
        self.lhs_str = lhs_str
        self.rhs_str = rhs_str
        self.lhs_name, self.lhs_indices = self._parse_term(lhs_str)
        self.rhs_terms = self._parse_rhs(rhs_str)

    def _parse_term(self, term_str):
        """Parses 'Name[i,j]' into ('Name', ['i', 'j'])"""
        match = re.match(r"(\w+)\[(.*?)\]", term_str.strip())
        if not match:
            raise ValueError(f"Invalid term format: {term_str}")
        name = match.group(1)
        indices = [idx.strip() for idx in match.group(2).split(',')]
        return name, indices

    def _parse_rhs(self, rhs_str):
        """
        Parses RHS. Supports simple term multiplication or a function wrapper.
        Examples: 
        - 'A[i,j] * B[j,k]'
        - 'step(A[i,j] * B[j,k])'
        - 'sigmoid(W[i,j] * X[j])'
        """
        rhs_str = rhs_str.strip()
        self.activation = None
        
        # Check for function wrapper
        match = re.match(r"(\w+)\((.*)\)", rhs_str)
        if match:
            func_name = match.group(1)
            if func_name in ['step', 'sigmoid']:
                self.activation = func_name
                rhs_str = match.group(2)
        
        terms_str = rhs_str.split('*')
        terms = []
        for t in terms_str:
            terms.append(self._parse_term(t))
        return terms

    def execute(self, context):
        """
        Executes the equation using einsum.
        :param context: Dictionary mapping tensor names to Tensor objects
        :return: A new Tensor object
        """
        # 1. Retrieve RHS tensors
        rhs_tensors = []
        input_subscripts = []
        for name, indices in self.rhs_terms:
            if name not in context:
                raise ValueError(f"Tensor {name} not found in context")
            tensor = context[name]
            rhs_tensors.append(tensor.data)
            input_subscripts.append("".join(indices))

        # 2. Construct einsum string
        # Format: "ij,jk->ik"
        lhs_subscript = "".join(self.lhs_indices)
        einsum_str = f"{','.join(input_subscripts)}->{lhs_subscript}"
        
        print(f"Executing: {self.lhs_str} = {self.rhs_str}")
        print(f"Einsum: {einsum_str}")

        # 3. Execute einsum
        result_data = np.einsum(einsum_str, *rhs_tensors)

        # 4. Apply nonlinearity
        if self.activation == 'step':
            result_data = (result_data > 0).astype(int)
        elif self.activation == 'sigmoid':
            result_data = 1 / (1 + np.exp(-result_data))
        
        return Tensor(self.lhs_name, self.lhs_indices, result_data)

class Program:
    def __init__(self):
        self.tensors = {}
        self.equations = []

    def add_tensor(self, tensor):
        self.tensors[tensor.name] = tensor

    def add_equation(self, equation_str):
        lhs, rhs = equation_str.split('=')
        eq = Equation(lhs.strip(), rhs.strip())
        self.equations.append(eq)

    def run(self):
        # Naive forward chaining: execute all equations once
        # In a real system, we'd loop until convergence or dependency order
        for eq in self.equations:
            result_tensor = eq.execute(self.tensors)
            # If it's a relation (heuristic: boolean inputs?), apply step
            # For this sample, let's just store the result
            self.tensors[result_tensor.name] = result_tensor
            print(f"Computed {result_tensor.name}:\n{result_tensor.data}")

def step(x):
    return (x > 0).astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
