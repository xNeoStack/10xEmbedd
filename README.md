# Inference Application

A TensorRT inference application with a GUI for selecting models and datasets, and displaying progress.

## Features

- Select TensorRT engine files (`.engine`).
- Choose a dataset folder with images.
- Visual progress bar and estimated time remaining.
- Supports custom batch sizes and threading options.

## Requirements

- Python 3.x
- TensorRT
- PyCUDA
- NumPy
- Pillow (PIL)
- Tkinter (usually included with Python)
- pynvml

## Installation

Install the required packages:

```bash
pip install numpy pycuda pillow pynvml

