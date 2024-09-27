### Reproducibility 

To ensure the reproducibility of our results, we have implemented several key measures:
- Random Seed Setting: We have set a fixed random seed for all stochastic processes involved in our experiments. This guarantees that our results can be consistently replicated when the code is run under the same conditions. Also, we have configured the TF\_CUDNN\_DETERMINISTIC flag in the OS environment. This ensures that the operations performed by TensorFlow are deterministic and repeatable, avoiding any non-deterministic behavior that could arise from the use of certain cuDNN kernels.
- Library Versions: We provide detailed information on the versions of all libraries and dependencies used in our experiments in the environment.yml file.
- Hardware Specifications: Our experiments were conducted using NVIDIA A100 GPUs with NVIDIA CUDA Version 12.2. 