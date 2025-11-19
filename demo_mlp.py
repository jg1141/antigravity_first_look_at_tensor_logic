from tensor_logic import Tensor, Program
import numpy as np

def run_demo():
    print("=== Tensor Logic MLP Demo ===")
    
    # Define dimensions
    input_size = 3
    output_size = 2
    
    # Input X [j]
    x_data = np.array([0.5, -0.2, 0.1])
    
    # Weights W [i, j] (output, input)
    # Random weights
    np.random.seed(42)
    w_data = np.random.randn(output_size, input_size)
    
    # Create Tensors
    X = Tensor("X", ["j"], x_data)
    W = Tensor("W", ["i", "j"], w_data)
    
    # Initialize Program
    prog = Program()
    prog.add_tensor(X)
    prog.add_tensor(W)
    
    # Define MLP Layer: Y[i] = sigmoid(W[i,j] * X[j])
    prog.add_equation("Y[i] = sigmoid(W[i,j] * X[j])")
    
    # Run Inference
    prog.run()
    
    # Check Result
    Y = prog.tensors["Y"]
    print("\nInput X:", X.data)
    print("Weights W:\n", W.data)
    print("Output Y:", Y.data)
    
    # Verify manually
    manual_y = 1 / (1 + np.exp(-np.dot(w_data, x_data)))
    print("\nManual Verification:", manual_y)
    assert np.allclose(Y.data, manual_y), "Mismatch between Tensor Logic and manual calculation!"
    print("Verification Successful!")

if __name__ == "__main__":
    run_demo()
