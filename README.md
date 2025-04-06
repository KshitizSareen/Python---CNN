# Convolutional Neural Network on MNIST (Parallel & Sequential)

This repository contains code for building and training a Convolutional Neural Network (CNN) on the MNIST dataset. The project demonstrates two versions of the same CNN architecture:

1. **CNN - Sequential**  
2. **CNN - Parallel** (using Python’s `multiprocessing`)

---

## Project Structure

The repository is organized into two main folders:

- **CNN - Sequential**  
  - `main.py`: Entry point for the sequential version of the CNN.
  
- **CNN - Parallel**  
  - `main.py`: Entry point for the parallel version of the CNN, leveraging Python’s `multiprocessing` to speed up computations.

Each folder contains any additional scripts, utility files, or modules required for that specific version of the implementation.

---

## CNN Architecture

Both the sequential and parallel versions use the same network architecture:

1. **Convolutional Layer** with 500 filters  
2. **2x2 Max Pooling Layer**  
3. **Convolutional Layer** with 500 filters  
4. **2x2 Max Pooling Layer**  
5. **Fully Connected Layer** with 1600 neurons  
6. **Fully Connected Layer** with 256 neurons  
7. **Fully Connected Layer** with 10 neurons (corresponding to 10 classes in MNIST)

**Training details** (as observed in tests on MNIST):

- Trained for **1000 epochs** with a learning rate of 0.001.
- **Initial training error** (epoch 1): **23.025835661401747**  
- **Final training error** (epoch 72): **0.00014451810090409874**

---

## Performance Comparison

- **Sequential Version**:  
  - **Average iteration time**: ~66 seconds

- **Parallel Version**:  
  - **Average iteration time**: ~20 seconds  
  - Achieves a significant speed-up by distributing computations across multiple processes.

---

## Dependencies

- [NumPy](https://numpy.org/) (for numerical computations)  
- Python 3.x

No additional libraries are strictly required for these demos.

---

## Getting Started

1. **Clone or Download** the repository.
2. Navigate to either the **CNN - Sequential** or **CNN - Parallel** folder.
3. Make sure you have [NumPy installed](https://numpy.org/install/):
    ```bash
    pip install numpy
    ```
4. Run the `main.py` script:
    ```bash
    python main.py
    ```
    This will start the training process on the MNIST dataset.

---

## Acknowledgments

- **MNIST Dataset**: [Yann LeCun’s MNIST Database](http://yann.lecun.com/exdb/mnist/)  
- **Python Multiprocessing**: Used in the parallel version to speed up convolutions and other operations.

---

Feel free to modify hyperparameters, number of epochs, or other training settings in the `main.py` file of each folder to explore different results. If you have any questions or issues, please open an issue or submit a pull request. Happy coding!
