# Kernel_Methods_Challenge

## Requirements

numpy<br>
torch (preferably with CUDA support)<br>
cvxpy (with OSQP support)

## Usage

`python run.py` to reproduce our best submission with public and private scores 67.13 and 67.87 respectively.

Additional scripts:<br>
-`main_MKL.py` to run the MKL optimization similar to SimpleMKL<br>
-`substring_kernel.py` to compute the substring kernel with dynamic programming