import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import threading
import os
from time import perf_counter
from queue import Queue
from PIL import Image
import glob

# Import TensorRT and PyCUDA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# For GPU usage monitoring
import pynvml

# Initialize PyCUDA and NVML
cuda.init()
pynvml.nvmlInit()

class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TensorRT Inference GUI")

        # Initialize ttk.Style
        self.style = ttk.Style()

        # Set default font
        default_font = ('Helvetica', 12)

        # Configure styles
        self.style.configure('.', font=default_font)

        # GUI Elements using ttk widgets
        # Model file selection (.onnx or .engine)
        self.model_label = ttk.Label(root, text="Select Model File (.onnx or .engine):")
        self.model_label.pack(pady=5)

        self.model_button = ttk.Button(root, text="Browse Model File", command=self.browse_model_file)
        self.model_button.pack(pady=5)

        self.model_path = tk.StringVar()
        self.model_entry = ttk.Entry(root, textvariable=self.model_path, width=60)
        self.model_entry.pack(pady=5)

        # Dataset folder selection
        self.dataset_label = ttk.Label(root, text="Select Dataset Folder:")
        self.dataset_label.pack(pady=5)

        self.dataset_button = ttk.Button(root, text="Browse Dataset Folder", command=self.browse_dataset_folder)
        self.dataset_button.pack(pady=5)

        self.dataset_path = tk.StringVar()
        self.dataset_entry = ttk.Entry(root, textvariable=self.dataset_path, width=60)
        self.dataset_entry.pack(pady=5)

        # Batch size selection
        self.batch_size_label = ttk.Label(root, text="Enter Batch Size:")
        self.batch_size_label.pack(pady=5)

        self.batch_size_var = tk.IntVar(value=32)
        self.batch_size_entry = ttk.Entry(root, textvariable=self.batch_size_var, width=10)
        self.batch_size_entry.pack(pady=5)

        # Number of pictures/entities per batch
        self.entities_per_batch_label = ttk.Label(root, text="Enter Number of Pictures per Batch:")
        self.entities_per_batch_label.pack(pady=5)

        self.entities_per_batch_var = tk.IntVar(value=32)
        self.entities_per_batch_entry = ttk.Entry(root, textvariable=self.entities_per_batch_var, width=10)
        self.entities_per_batch_entry.pack(pady=5)

        # Queue size selection
        self.queue_size_label = ttk.Label(root, text="Enter Queue Size:")
        self.queue_size_label.pack(pady=5)

        self.queue_size_var = tk.IntVar(value=10)
        self.queue_size_entry = ttk.Entry(root, textvariable=self.queue_size_var, width=10)
        self.queue_size_entry.pack(pady=5)

        # Number of working threads selection
        self.threads_label = ttk.Label(root, text="Enter Number of Working Threads:")
        self.threads_label.pack(pady=5)

        self.threads_var = tk.IntVar(value=4)
        self.threads_entry = ttk.Entry(root, textvariable=self.threads_var, width=10)
        self.threads_entry.pack(pady=5)

        # Number of CUDA streams selection
        self.streams_label = ttk.Label(root, text="Enter Number of CUDA Streams:")
        self.streams_label.pack(pady=5)

        self.streams_var = tk.IntVar(value=1)
        self.streams_entry = ttk.Entry(root, textvariable=self.streams_var, width=10)
        self.streams_entry.pack(pady=5)

        # Perform Inference Button
        self.inference_button = ttk.Button(root, text="Perform Inference", command=self.run_inference)
        self.inference_button.pack(pady=20)

        # Progress bar
        self.progress = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress, orient=tk.HORIZONTAL, length=400, mode='determinate', maximum=100)
        self.progress_bar.pack(pady=10)

        # Textbox to display inference output result
        self.output_text = tk.Text(root, height=10, width=80)
        self.output_text.pack(pady=10)

        # Variables
        self.engine = None
        self.dataset_images = []
        self.num_batches_processed = 0
        self.total_batches = 0
        self.start_time = None

        # For GPU usage monitoring
        self.gpu_usages = []
        self.gpu_usage_lock = threading.Lock()  # To ensure thread-safe access

    def browse_model_file(self):
        model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.onnx *.engine")])
        self.model_path.set(model_path)

    def browse_dataset_folder(self):
        dataset_folder = filedialog.askdirectory()
        self.dataset_path.set(dataset_folder)

    def build_engine_from_onnx(self, onnx_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30  # Adjust as necessary
            builder.max_batch_size = self.batch_size_var.get()

            # Parse ONNX model
            with open(onnx_file_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    error_msgs = ''
                    for error in range(parser.num_errors):
                        error_msgs += f'{parser.get_error(error)}\n'
                    raise RuntimeError(f"Failed to parse ONNX model:\n{error_msgs}")

            # Build the engine
            self.output_text.insert(tk.END, "Building TensorRT engine from ONNX model...\n")
            self.root.update_idletasks()
            engine = builder.build_cuda_engine(network)
            if engine is None:
                raise RuntimeError("Failed to build the TensorRT engine from ONNX model.")
            return engine

    def load_engine(self, model_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file not found. Please check the path.")
            return None

        try:
            if model_path.endswith('.engine'):
                # Load TensorRT engine directly
                with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                    engine = runtime.deserialize_cuda_engine(f.read())
                    if engine is None:
                        raise RuntimeError("Failed to load the TensorRT engine.")
                    return engine
            elif model_path.endswith('.onnx'):
                # Build TensorRT engine from ONNX model
                engine = self.build_engine_from_onnx(model_path)
                return engine
            else:
                messagebox.showerror("Error", "Unsupported model format. Please select a .onnx or .engine file.")
                return None
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading the model:\n{str(e)}")
            return None

    def load_images_from_folder(self, folder_path):
        # Supported image extensions
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        images = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale for MNIST
                img = img.resize((28, 28))  # Resize to MNIST image size
                img_array = np.asarray(img).astype(np.float32)
                img_array = img_array / 255.0  # Normalize pixel values
                img_array = img_array[np.newaxis, :, :]  # Add channel dimension
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        return images

    def run_inference(self):
        model_path = self.model_path.get()
        dataset_folder = self.dataset_path.get()
        batch_size = self.batch_size_var.get()
        entities_per_batch = self.entities_per_batch_var.get()
        queue_size = self.queue_size_var.get()

        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file (.onnx or .engine).")
            return

        if not os.path.exists(dataset_folder):
            messagebox.showerror("Error", "Please select a valid dataset folder.")
            return

        if batch_size <= 0 or queue_size <= 0 or entities_per_batch <= 0:
            messagebox.showerror("Error", "Please enter valid values for batch size, queue size, and entities per batch.")
            return

        # Load images
        self.output_text.insert(tk.END, "Loading images...\n")
        self.root.update_idletasks()
        self.dataset_images = self.load_images_from_folder(dataset_folder)
        if len(self.dataset_images) == 0:
            messagebox.showerror("Error", "No valid images found in the selected folder.")
            return

        # Adjust batch size based on entities per batch
        batch_size = entities_per_batch
        self.batch_size_var.set(batch_size)

        # Calculate total batches
        self.total_batches = (len(self.dataset_images) + batch_size - 1) // batch_size

        # Load the TensorRT engine
        self.engine = self.load_engine(model_path)
        if self.engine is None:
            messagebox.showerror("Error", "Failed to load the model. Cannot proceed with inference.")
            return

        # Initialize the queue with the user-specified size
        self.batch_queue = Queue(maxsize=queue_size)
        threading.Thread(target=self.enqueue_batches).start()

        # Start inference in separate threads
        num_threads = self.threads_var.get()
        self.start_time = perf_counter()  # Start time for total inference
        self.num_batches_processed = 0  # Reset batch counter
        for _ in range(num_threads):
            threading.Thread(target=self.inference_worker).start()

    def enqueue_batches(self):
        batch_size = self.batch_size_var.get()
        for batch_start in range(0, len(self.dataset_images), batch_size):
            batch_images = self.dataset_images[batch_start:batch_start + batch_size]
            self.batch_queue.put(batch_images)
            print(f"Enqueued batch {batch_start // batch_size + 1}")

    def get_gpu_usage(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu

    def get_output_shape(self, binding_idx, context):
        # Since we cannot retrieve the binding shape, we will define it manually
        batch_size = self.batch_size_var.get()
        output_size = self.get_output_size(binding_idx)
        output_shape = (batch_size, output_size)
        return output_shape

    def get_output_size(self, binding_idx):
        # You need to specify the output size based on your model
        # For MNIST, the output size is 10
        return 10  # Replace with your model's actual output size

    def inference_worker(self):
        batch_size = self.batch_size_var.get()
        num_streams = self.streams_var.get()
        device = cuda.Device(0)
        context = device.make_context()
        try:
            streams = [cuda.Stream() for _ in range(num_streams)]
            execution_context = self.engine.create_execution_context()

            while True:
                try:
                    batch_images = self.batch_queue.get(timeout=5)
                except:
                    break  # Queue is empty

                # Preprocess batch
                batch_array = np.array(batch_images, dtype=np.float32)
                # Reshape to (batch_size, 1, 28, 28)
                batch_array = batch_array.reshape(batch_size, 1, 28, 28)

                # Ensure the batch size matches the expected input
                if batch_array.shape[0] < batch_size:
                    padding = ((0, batch_size - batch_array.shape[0]), (0, 0), (0, 0), (0, 0))
                    batch_array = np.pad(batch_array, padding, mode='constant', constant_values=0)

                # Set input binding shape if dynamic
                try:
                    execution_context.set_binding_shape(0, batch_array.shape)
                except AttributeError:
                    pass  # Proceed if set_binding_shape is not available

                # Assuming data type as float32
                input_dtype = np.float32
                batch_array = batch_array.astype(input_dtype)

                # Transfer input data to the GPU.
                d_input = cuda.mem_alloc(batch_array.nbytes)
                cuda.memcpy_htod(d_input, batch_array)

                # Allocate memory for output
                output_shape = self.get_output_shape(1, execution_context)

                # Assuming data type as float32
                output_dtype = np.float32
                h_output = np.empty(output_shape, dtype=output_dtype)
                d_output = cuda.mem_alloc(h_output.nbytes)

                bindings = [int(d_input), int(d_output)]

                # Get GPU usage before inference
                gpu_usage_before = self.get_gpu_usage()

                # Run inference
                execution_context.execute_v2(bindings)

                # Get GPU usage after inference
                gpu_usage_after = self.get_gpu_usage()
                avg_gpu_usage = (gpu_usage_before + gpu_usage_after) / 2

                # Collect GPU usage data
                with self.gpu_usage_lock:
                    self.gpu_usages.append(avg_gpu_usage)

                # Transfer predictions back from the GPU.
                cuda.memcpy_dtoh(h_output, d_output)

                # Process output (here we just print the batch number)
                with threading.Lock():
                    self.num_batches_processed += 1
                    progress_percent = (self.num_batches_processed / self.total_batches) * 100
                    self.progress.set(progress_percent)

                self.output_text.insert(tk.END, f"Inference completed for batch {self.num_batches_processed}\n")
                self.root.update_idletasks()

            # Update progress to 100% when done
            if self.num_batches_processed >= self.total_batches:
                self.progress.set(100)
                self.output_text.insert(tk.END, "Inference completed for all batches.\n")
                self.root.update_idletasks()
                self.finalize_inference()

        except Exception as e:
            print(f"Error during inference: {e}")
            self.output_text.insert(tk.END, f"Error during inference: {e}\n")
        finally:
            context.pop()

    def finalize_inference(self):
        total_inference_time = perf_counter() - self.start_time

        # Calculate average GPU usage
        if self.gpu_usages:
            avg_gpu_usage = sum(self.gpu_usages) / len(self.gpu_usages)
        else:
            avg_gpu_usage = 0

        # Display metrics
        metrics = f"\n==== Inference Summary ====\n" \
                  f"Total Inference Time: {total_inference_time:.4f} seconds\n" \
                  f"Average GPU Usage: {avg_gpu_usage:.2f}%\n"

        self.output_text.insert(tk.END, metrics)
        print(metrics)

# Main function to launch the GUI
def main():
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
