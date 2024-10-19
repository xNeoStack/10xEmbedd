import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import threading
import os
from time import perf_counter
import pynvml
from queue import Queue

# Initialize PyCUDA and NVML for GPU usage tracking
cuda.init()
pynvml.nvmlInit()

class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TensorRT Inference GUI")

        # Engine file selection
        self.engine_label = tk.Label(root, text="Select TensorRT Engine (.engine):")
        self.engine_label.pack(pady=5)

        self.engine_button = tk.Button(root, text="Browse Engine File", command=self.browse_engine_file)
        self.engine_button.pack(pady=5)

        self.engine_path = tk.StringVar()
        self.engine_entry = tk.Entry(root, textvariable=self.engine_path, width=50)
        self.engine_entry.pack(pady=5)

        # Dataset folder selection
        self.dataset_label = tk.Label(root, text="Select Dataset Folder:")
        self.dataset_label.pack(pady=5)

        self.dataset_button = tk.Button(root, text="Browse Dataset Folder", command=self.browse_dataset_folder)
        self.dataset_button.pack(pady=5)

        self.dataset_path = tk.StringVar()
        self.dataset_entry = tk.Entry(root, textvariable=self.dataset_path, width=50)
        self.dataset_entry.pack(pady=5)

        # Input shape and data type selection
        self.input_shape_label = tk.Label(root, text="Enter input shape (e.g., 1,28,28):")
        self.input_shape_label.pack(pady=5)

        self.input_shape_var = tk.StringVar(value="1,28,28")
        self.input_shape_entry = tk.Entry(root, textvariable=self.input_shape_var, width=20)
        self.input_shape_entry.pack(pady=5)

        self.data_type_label = tk.Label(root, text="Select data type:")
        self.data_type_label.pack(pady=5)

        self.data_type_var = tk.StringVar(value="float32")
        self.data_type_menu = tk.OptionMenu(root, self.data_type_var, "float32", "float16", "int8")
        self.data_type_menu.pack(pady=5)

        # Batch size selection
        self.batch_size_label = tk.Label(root, text="Enter number of images per batch:")
        self.batch_size_label.pack(pady=5)

        self.batch_size_var = tk.IntVar(value=32)
        self.batch_size_entry = tk.Entry(root, textvariable=self.batch_size_var, width=10)
        self.batch_size_entry.pack(pady=5)

        # Total number of batches selection
        self.num_batches_label = tk.Label(root, text="Enter number of batches:")
        self.num_batches_label.pack(pady=5)

        self.num_batches_var = tk.IntVar(value=10)
        self.num_batches_entry = tk.Entry(root, textvariable=self.num_batches_var, width=10)
        self.num_batches_entry.pack(pady=5)

        # FIFO queue size selection
        self.queue_size_label = tk.Label(root, text="Enter queue size if FIFO is enabled:")
        self.queue_size_label.pack(pady=5)

        self.queue_size_var = tk.IntVar(value=40)
        self.queue_size_entry = tk.Entry(root, textvariable=self.queue_size_var, width=10)
        self.queue_size_entry.pack(pady=5)

        # Checkbox to enable or disable FIFO queue
        self.fifo_enabled = tk.BooleanVar(value=True)
        self.fifo_checkbox = tk.Checkbutton(root, text="Enable FIFO (Batch Queue)", variable=self.fifo_enabled)
        self.fifo_checkbox.pack(pady=10)

        # Number of working threads selection
        self.threads_label = tk.Label(root, text="Enter number of working threads:")
        self.threads_label.pack(pady=5)

        self.threads_var = tk.IntVar(value=4)
        self.threads_entry = tk.Entry(root, textvariable=self.threads_var, width=10)
        self.threads_entry.pack(pady=5)

        # Number of CUDA streams selection
        self.streams_label = tk.Label(root, text="Enter number of CUDA streams:")
        self.streams_label.pack(pady=5)

        self.streams_var = tk.IntVar(value=26)
        self.streams_entry = tk.Entry(root, textvariable=self.streams_var, width=10)
        self.streams_entry.pack(pady=5)

        # Perform Inference Button
        self.inference_button = tk.Button(root, text="Perform Inference", command=self.run_inference)
        self.inference_button.pack(pady=20)

        # Textbox to display inference output result
        self.output_text = tk.Text(root, height=10, width=50)
        self.output_text.pack(pady=10)

        # Progress bar for inference
        self.progress = ttk.Progressbar(root, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(pady=10)

        # FIFO Queue
        self.batch_queue = None
        self.queue_lock = threading.Lock()
        self.engine = None

        # Data for logging
        self.start_time = None
        self.batch_times = []
        self.gpu_usages = []
        self.num_batches_processed = 0

    def browse_engine_file(self):
        engine_path = filedialog.askopenfilename(filetypes=[("TensorRT Engine", "*.engine")])
        self.engine_path.set(engine_path)

    def browse_dataset_folder(self):
        dataset_path = filedialog.askdirectory()
        self.dataset_path.set(dataset_path)

    def load_dataset(self):
        dataset_folder = self.dataset_path.get()
        if not os.path.exists(dataset_folder):
            messagebox.showerror("Error", "Dataset folder not found. Please check the path.")
            return None

        input_shape = tuple(map(int, self.input_shape_var.get().split(',')))
        data_type = np.dtype(self.data_type_var.get())

        # Placeholder for actual dataset loading logic
        print("Loading dataset from folder:", dataset_folder)
        return np.random.rand(1000, *input_shape).astype(data_type)

    def load_engine(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        if not os.path.exists(engine_path):
            messagebox.showerror("Error", "Engine file not found. Please check the path.")
            return None

        try:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    raise RuntimeError("Failed to load the TensorRT engine.")
                return engine
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading the engine:\n{str(e)}")
            return None

    def run_inference(self):
        engine_path = self.engine_path.get()

        if not os.path.exists(engine_path):
            messagebox.showerror("Error", "Please select a valid .engine file.")
            return

        num_batches = self.num_batches_var.get()
        batch_size = self.batch_size_var.get()
        queue_size = self.queue_size_var.get()

        if num_batches <= 0 or batch_size <= 0 or queue_size <= 0:
            messagebox.showerror("Error", "Please enter valid values for batch size, number of batches, and queue size.")
            return

        self.start_time = perf_counter()

        digit_sequence = self.load_dataset()[:num_batches * batch_size]

        self.engine = self.load_engine(engine_path)

        if self.engine is None:
            messagebox.showerror("Error", "Failed to load the TensorRT engine. Cannot proceed with inference.")
            return

        num_threads = self.threads_var.get()
        num_streams = self.streams_var.get()

        if self.fifo_enabled.get():
            print(f"FIFO is enabled with queue size {queue_size}. Using batch queue.")
            self.batch_queue = Queue(maxsize=queue_size)
            threading.Thread(target=self.enqueue_batches, args=(digit_sequence, batch_size)).start()
            for _ in range(num_streams):
                threading.Thread(target=self.inference_worker_with_fifo, args=(batch_size,)).start()
        else:
            print("FIFO is disabled. Running inference directly.")
            threading.Thread(target=self.inference_worker_without_fifo, args=(digit_sequence, batch_size)).start()

        self.progress['maximum'] = num_batches

    def enqueue_batches(self, digit_sequence, batch_size):
        device = cuda.Device(0)
        context = device.make_context()
        try:
            for i in range(0, len(digit_sequence), batch_size):
                self.batch_queue.put((i, digit_sequence[i:i + batch_size]), block=True)
                print(f"Batch {i // batch_size} enqueued")
        finally:
            context.pop()

    def inference_worker_with_fifo(self, batch_size):
        device = cuda.Device(0)
        context = device.make_context()
        try:
            streams = [cuda.Stream() for _ in range(self.streams_var.get())]
            execution_context = self.engine.create_execution_context()

            while not self.batch_queue.empty():
                batch_index, batch_data = self.batch_queue.get(block=True)
                with self.queue_lock:
                    batch_data = np.ascontiguousarray(batch_data.reshape(batch_size, *map(int, self.input_shape_var.get().split(','))))

                    d_input = cuda.mem_alloc(batch_data.nbytes)

                    cuda.memcpy_htod_async(d_input, batch_data, streams[batch_index % len(streams)])

                    gpu_usage_before = self.get_gpu_usage()
                    batch_start_time = perf_counter()

                    output = self.perform_inference(execution_context, d_input, batch_size, streams[batch_index % len(streams)])

                    batch_end_time = perf_counter()
                    gpu_usage_after = self.get_gpu_usage()

                    batch_runtime = batch_end_time - batch_start_time
                    avg_gpu_usage = (gpu_usage_before + gpu_usage_after) / 2

                    self.batch_times.append(batch_runtime)
                    self.gpu_usages.append(avg_gpu_usage)

                    self.output_text.insert(tk.END, f"Inference completed for batch {batch_index // batch_size}\n")
                    print(f"Inference completed for batch {batch_index // batch_size}")

                    self.num_batches_processed += 1
                    self.progress['value'] = self.num_batches_processed

        except Exception as e:
            print(f"Error in worker thread: {e}")
        finally:
            context.pop()
            self.finalize_inference()

    def inference_worker_without_fifo(self, digit_sequence, batch_size):
        device = cuda.Device(0)
        context = device.make_context()
        try:
            stream = cuda.Stream()
            execution_context = self.engine.create_execution_context()

            for i in range(0, len(digit_sequence), batch_size):
                batch_data = digit_sequence[i:i + batch_size]
                batch_data = np.ascontiguousarray(batch_data.reshape(batch_size, *map(int, self.input_shape_var.get().split(','))))

                d_input = cuda.mem_alloc(batch_data.nbytes)

                cuda.memcpy_htod_async(d_input, batch_data, stream)

                gpu_usage_before = self.get_gpu_usage()
                batch_start_time = perf_counter()

                output = self.perform_inference(execution_context, d_input, batch_size, stream)

                batch_end_time = perf_counter()
                gpu_usage_after = self.get_gpu_usage()

                batch_runtime = batch_end_time - batch_start_time
                avg_gpu_usage = (gpu_usage_before + gpu_usage_after) / 2

                self.batch_times.append(batch_runtime)
                self.gpu_usages.append(avg_gpu_usage)

                self.output_text.insert(tk.END, f"Inference completed for batch {i // batch_size}\n")
                print(f"Inference completed for batch {i // batch_size}")

                self.num_batches_processed += 1
                self.progress['value'] = self.num_batches_processed

        except Exception as e:
            print(f"Error in worker thread: {e}")
        finally:
            context.pop()
            self.finalize_inference()

    def perform_inference(self, context, d_input, batch_size, stream):
        input_shape = tuple(map(int, self.input_shape_var.get().split(',')))
        data_type = np.dtype(self.data_type_var.get())
        output_size = batch_size * np.prod(input_shape) * data_type.itemsize
        d_output = cuda.mem_alloc(output_size)

        bindings = [int(d_input), int(d_output)]
        context.execute_v2(bindings)

        output = np.empty((batch_size, *input_shape), dtype=data_type)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        d_output.free()
        return output

    def get_gpu_usage(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu

    def finalize_inference(self):
        total_inference_time = perf_counter() - self.start_time

        if self.num_batches_processed > 0:
            avg_batch_time = sum(self.batch_times) / self.num_batches_processed
        else:
            avg_batch_time = 0

        if self.gpu_usages:
            avg_gpu_usage = sum(self.gpu_usages) / len(self.gpu_usages)
            max_gpu_usage = max(self.gpu_usages)
            min_gpu_usage = min(self.gpu_usages)
        else:
            avg_gpu_usage = max_gpu_usage = min_gpu_usage = 0

        print("\n==== Inference Summary ====")
        print(f"Model Name: {os.path.basename(self.engine_path.get())}")
        print(f"Number of Pictures per Batch: {self.batch_size_var.get()}")
        print(f"Number of Batches: {self.num_batches_var.get()}")
        if self.fifo_enabled.get():
            print(f"FIFO Queue Size: {self.queue_size_var.get()}")
        print(f"Min GPU Load: {min_gpu_usage:.2f}%")
        print(f"Max GPU Load: {max_gpu_usage:.2f}%")
        print(f"Avg GPU Load: {avg_gpu_usage:.2f}%")
        print(f"Total Inference Time: {total_inference_time:.4f} seconds")
        print(f"Avg Time per Batch: {avg_batch_time:.4f} seconds")

def main():
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()