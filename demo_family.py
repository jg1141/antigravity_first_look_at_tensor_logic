from tensor_logic import Tensor, Program
import numpy as np

def run_demo():
    print("=== Tensor Logic Family Relations Demo ===")
    
    # Define individuals
    # 0: Ann, 1: Bob, 2: Charlie, 3: Dan
    names = {0: "Ann", 1: "Bob", 2: "Charlie", 3: "Dan"}
    
    # Define relations
    # Sister(x, y): x is sister of y
    # Ann (0) is sister of Bob (1)
    sister_data = np.zeros((4, 4), dtype=int)
    sister_data[0, 1] = 1
    
    # Parent(y, z): y is parent of z
    # Bob (1) is parent of Charlie (2)
    parent_data = np.zeros((4, 4), dtype=int)
    parent_data[1, 2] = 1
    
    # Create Tensors
    sister = Tensor("Sister", ["x", "y"], sister_data)
    parent = Tensor("Parent", ["y", "z"], parent_data)
    
    # Initialize Program
    prog = Program()
    prog.add_tensor(sister)
    prog.add_tensor(parent)
    
    # Define Rule: Aunt(x, z) :- Sister(x, y), Parent(y, z)
    # In Tensor Logic: Aunt[x,z] = step(Sister[x,y] * Parent[y,z])
    prog.add_equation("Aunt[x,z] = step(Sister[x,y] * Parent[y,z])")
    
    # Run Inference
    prog.run()
    
    # Check Result
    aunt = prog.tensors["Aunt"]
    print("\nAunt Relation Matrix:")
    print(aunt.data)
    
    # Interpret results
    indices = np.argwhere(aunt.data > 0)
    for x, z in indices:
        print(f"\nResult: {names[x]} is the Aunt of {names[z]}")

if __name__ == "__main__":
    run_demo()
