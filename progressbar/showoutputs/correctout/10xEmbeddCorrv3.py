import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import threading
import os
from time import perf_counter
from queue import Queue, Empty
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Import TensorRT and PyCUDA
import tensorrt as trt
import pycuda.driver as cuda

# For GPU usage monitoring
import pynvml
import time

# Initialize NVML for GPU monitoring
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
        self.output_text = tk.Text(root, height=10, width=80, state='disabled')
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
        self.inference_running = False  # Flag to control GPU monitoring thread

        # Thread-safe queue for GUI updates
        self.gui_queue = Queue()

        # Start GUI update loop
        self.root.after(100, self.process_gui_queue)

    def log_output(self, message):
        """Thread-safe method to log messages to the output_text widget."""
        self.gui_queue.put(message)

    def process_gui_queue(self):
        """Process messages from the gui_queue and update the output_text widget."""
        try:
            while True:
                message = self.gui_queue.get_nowait()
                self.output_text.configure(state='normal')
                self.output_text.insert(tk.END, message + "\n")
                self.output_text.see(tk.END)
                self.output_text.configure(state='disabled')
        except Empty:
            pass
        finally:
            self.root.after(100, self.process_gui_queue)

    def browse_model_file(self):
        model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.onnx *.engine")])
        if model_path:
            self.model_path.set(model_path)
            self.log_output(f"Selected model file: {model_path}")

    def browse_dataset_folder(self):
        dataset_folder = filedialog.askdirectory()
        if dataset_folder:
            self.dataset_path.set(dataset_folder)
            self.log_output(f"Selected dataset folder: {dataset_folder}")

    def build_engine_from_onnx(self, onnx_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags=network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Define optimization profiles for dynamic shapes if needed
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 if supported

        # Parse ONNX model
        with open(onnx_file_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                error_msgs = ''
                for error in range(parser.num_errors):
                    error_msgs += f'{parser.get_error(error)}\n'
                raise RuntimeError(f"Failed to parse ONNX model:\n{error_msgs}")

        # Log all binding names for verification
        binding_names = [network.get_input(i).name for i in range(network.num_inputs)]
        binding_names += [network.get_output(i).name for i in range(network.num_outputs)]
        self.log_output(f"Model bindings: {binding_names}")

        # Build the engine
        self.log_output("Building TensorRT engine from ONNX model...")
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build the TensorRT engine from ONNX model.")
        self.log_output("TensorRT engine successfully built.")
        return engine

    def load_engine(self, model_path):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

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
                    self.log_output("TensorRT engine successfully loaded from .engine file.")
                    
                    # Log all binding names for verification
                    binding_names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
                    self.log_output(f"Engine bindings: {binding_names}")
                    
                    return engine
            elif model_path.endswith('.onnx'):
                # Build TensorRT engine from ONNX model
                engine = self.build_engine_from_onnx(model_path)
                
                # Log all binding names for verification
                binding_names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
                self.log_output(f"Engine bindings after building: {binding_names}")
                
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
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((28, 28))  # Resize to model's expected size
                img_array = np.asarray(img).astype(np.float32)
                img_array = img_array / 255.0  # Normalize pixel values
                img_array = img_array[np.newaxis, :, :]  # Add channel dimension (1, 28, 28)
                images.append(img_array)
            except Exception as e:
                self.log_output(f"Error loading image {img_path}: {e}")
        self.log_output(f"Loaded {len(images)} images from dataset.")
        return images

    def run_inference(self):
        model_path = self.model_path.get()
        dataset_folder = self.dataset_path.get()
        batch_size = self.batch_size_var.get()
        entities_per_batch = self.entities_per_batch_var.get()
        queue_size = self.queue_size_var.get()

        # Input Validation
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file (.onnx or .engine).")
            return

        if not os.path.exists(dataset_folder):
            messagebox.showerror("Error", "Please select a valid dataset folder.")
            return

        if batch_size <= 0 or queue_size <= 0 or entities_per_batch <= 0:
            messagebox.showerror("Error", "Please enter valid values for batch size, queue size, and entities per batch.")
            return

        # Disable the inference button to prevent multiple inferences
        self.inference_button.config(state='disabled')

        # Start inference in a separate thread to keep GUI responsive
        threading.Thread(target=self.perform_inference, daemon=True).start()

    def perform_inference(self):
        try:
            # Load images
            self.log_output("Loading images...")
            self.dataset_images = self.load_images_from_folder(self.dataset_path.get())
            if len(self.dataset_images) == 0:
                messagebox.showerror("Error", "No valid images found in the selected folder.")
                self.inference_button.config(state='normal')
                return

            # Adjust batch size based on entities per batch
            batch_size = self.entities_per_batch_var.get()
            self.batch_size_var.set(batch_size)

            # Calculate total batches
            self.total_batches = (len(self.dataset_images) + batch_size - 1) // batch_size
            self.log_output(f"Total batches to process: {self.total_batches}")

            # Load the TensorRT engine
            self.engine = self.load_engine(self.model_path.get())
            if self.engine is None:
                messagebox.showerror("Error", "Failed to load the model. Cannot proceed with inference.")
                self.inference_button.config(state='normal')
                return

            # Initialize the queue with the user-specified size
            self.batch_queue = Queue(maxsize=self.queue_size_var.get())
            threading.Thread(target=self.enqueue_batches, daemon=True).start()

            # Start GPU usage monitoring
            self.inference_running = True
            threading.Thread(target=self.monitor_gpu_usage, daemon=True).start()

            # Start inference in separate threads
            num_threads = self.threads_var.get()
            self.start_time = perf_counter()  # Start time for total inference
            self.num_batches_processed = 0  # Reset batch counter
            for i in range(num_threads):
                threading.Thread(target=self.inference_worker, daemon=True, name=f"InferenceWorker-{i+1}").start()

        except Exception as e:
            self.log_output(f"Error during inference setup: {e}")
            messagebox.showerror("Error", f"An error occurred during inference setup:\n{str(e)}")
            self.inference_button.config(state='normal')

    def enqueue_batches(self):
        batch_size = self.batch_size_var.get()
        for batch_start in range(0, len(self.dataset_images), batch_size):
            batch_images = self.dataset_images[batch_start:batch_start + batch_size]
            self.batch_queue.put(batch_images)
            self.log_output(f"Enqueued batch {batch_start // batch_size + 1}")
        self.log_output("All batches enqueued.")

    def monitor_gpu_usage(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        while self.inference_running:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            with self.gpu_usage_lock:
                self.gpu_usages.append(util.gpu)
            time.sleep(0.1)  # Sample every 100ms

    def get_binding_index_safe(self, binding_name, default_idx):
        """
        Safely retrieve the binding index by name.
        If the binding does not exist, return the default index.
        """
        try:
            return self.engine.get_binding_index(binding_name)
        except trt.RuntimeError:
            self.log_output(f"Binding '{binding_name}' not found. Using default binding index {default_idx}.")
            return default_idx

    def get_output_shape(self, context, binding_idx):
        """Retrieve the output shape from the TensorRT context."""
        binding_shape = context.get_binding_shape(binding_idx)
        return tuple(binding_shape)

    def inference_worker(self):
        # Each worker thread needs its own CUDA context
        try:
            # Create CUDA context for this thread
            context = cuda.Device(0).make_context()
            self.log_output(f"{threading.current_thread().name}: CUDA context created.")

            # Create execution context
            execution_context = self.engine.create_execution_context()

            while True:
                try:
                    batch_images = self.batch_queue.get(timeout=5)
                except Empty:
                    self.log_output(f"{threading.current_thread().name}: No more batches to process.")
                    break  # Queue is empty

                # Preprocess batch
                batch_array = np.concatenate(batch_images, axis=0)  # Shape: (batch_size, 1, 28, 28)
                actual_batch_size = batch_array.shape[0]

                # Retrieve binding indices
                input_binding_idx = self.get_binding_index_safe("input", 0)
                output_binding_idx = self.get_binding_index_safe("output", 1)

                input_shape = self.engine.get_binding_shape(input_binding_idx)
                output_shape = self.get_output_shape(execution_context, output_binding_idx)

                # Adjust shapes if dynamic
                if -1 in input_shape:
                    try:
                        execution_context.set_binding_shape(input_binding_idx, batch_array.shape)
                        input_shape = self.engine.get_binding_shape(input_binding_idx)
                    except Exception as e:
                        self.log_output(f"{threading.current_thread().name}: Error setting binding shape: {e}")
                        continue

                # Calculate size of input and output
                input_size = trt.volume(input_shape) * self.engine.max_batch_size
                output_size = trt.volume(output_shape) * self.engine.max_batch_size

                # Allocate GPU memory
                d_input = cuda.mem_alloc(batch_array.nbytes)
                h_output = np.empty(output_shape, dtype=np.float32)
                d_output = cuda.mem_alloc(h_output.nbytes)

                # Transfer input data to the GPU.
                cuda.memcpy_htod(d_input, batch_array)

                # Set bindings
                bindings = [int(d_input), int(d_output)]

                # Run inference
                try:
                    execution_context.execute_v2(bindings=bindings)
                except Exception as e:
                    self.log_output(f"{threading.current_thread().name}: Inference execution failed: {e}")
                    # Free memory and continue
                    d_input.free()
                    d_output.free()
                    continue

                # Transfer predictions back from the GPU.
                cuda.memcpy_dtoh(h_output, d_output)

                # Process output: Extract predicted classes
                predicted_classes = np.argmax(h_output, axis=1)

                # Log predictions
                with threading.Lock():
                    self.num_batches_processed += 1
                    progress_percent = (self.num_batches_processed / self.total_batches) * 100
                    self.gui_queue.put(f"Processed batch {self.num_batches_processed}/{self.total_batches}")
                    self.progress.set(progress_percent)

                # Optionally visualize a subset of predictions
                if self.num_batches_processed % max(1, self.total_batches // 10) == 0:
                    self.visualize_predictions(batch_images, predicted_classes, self.num_batches_processed)

                # Clean up
                d_input.free()
                d_output.free()

            # After processing all batches
            if self.num_batches_processed >= self.total_batches:
                self.inference_running = False
                self.finalize_inference()

        except Exception as e:
            self.log_output(f"{threading.current_thread().name}: Error during inference: {e}")
            self.inference_running = False
            messagebox.showerror("Error", f"An error occurred during inference:\n{str(e)}")
        finally:
            # Clean up the CUDA context
            try:
                context.pop()
                del context
                self.log_output(f"{threading.current_thread().name}: CUDA context destroyed.")
            except Exception as e:
                self.log_output(f"{threading.current_thread().name}: Error destroying CUDA context: {e}")

    def visualize_predictions(self, batch_images, predictions, batch_number, num=5):
        """Visualize a subset of images and their predictions."""
        try:
            images = [img.reshape(28, 28) for img in batch_images[:num]]
            plt.figure(figsize=(15, 3 * num))
            for i in range(num):
                plt.subplot(1, num, i + 1)
                plt.imshow(images[i], cmap='gray')
                plt.title(f'Batch {batch_number}\nPred: {predictions[i]}')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.log_output(f"Error during visualization: {e}")

    def get_average_gpu_usage(self):
        """Calculate the average GPU usage."""
        with self.gpu_usage_lock:
            if len(self.gpu_usages) == 0:
                return 0
            return sum(self.gpu_usages) / len(self.gpu_usages)

    def finalize_inference(self):
        total_inference_time = perf_counter() - self.start_time

        # Calculate average GPU usage
        avg_gpu_usage = self.get_average_gpu_usage()

        # Display metrics
        metrics = (
            "\n==== Inference Summary ====\n"
            f"Total Inference Time: {total_inference_time:.4f} seconds\n"
            f"Average GPU Usage: {avg_gpu_usage:.2f}%\n"
        )

        self.log_output(metrics)
        self.log_output("Inference completed for all batches.")

        # Re-enable the inference button
        self.inference_button.config(state='normal')


# Main function to launch the GUI
def main():
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
