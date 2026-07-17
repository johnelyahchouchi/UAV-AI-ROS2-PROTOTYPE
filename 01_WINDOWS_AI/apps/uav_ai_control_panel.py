import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, simpledialog
import subprocess
import threading
import queue
import sys
from pathlib import Path


PROJECT_DIR = Path(r"C:\Users\UAVlab\Desktop\uav_ai_company")
SENDER_SCRIPT = PROJECT_DIR / "win_yolo_tcp_sender.py"


MODES = {
    "General surveillance - vehicles.mp4": {
        "model": "yolov8n.pt",
        "source": "vehicles.mp4",
        "conf": "0.25"
    },
    "BTR armored - real tank video": {
        "model": "btr_best_v2.pt",
        "source": "tank_real_test.mp4",
        "conf": "0.4"
    },
    "BTR armored - demo video": {
        "model": "btr_best_v2.pt",
        "source": "btr_demo.mp4",
        "conf": "0.4"
    },
    "Live stream - general surveillance": {
        "model": "yolov8n.pt",
        "source": "",
        "conf": "0.25"
    },
    "Custom source / custom model": {
        "model": "",
        "source": "",
        "conf": "0.25"
    }
}


class UAVAIControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("UAV AI + ROS2 Control Panel")
        self.root.geometry("900x620")

        self.process = None
        self.log_queue = queue.Queue()

        self.vm_ip = tk.StringVar(value="192.168.153.128")
        self.mode = tk.StringVar(value="BTR armored - real tank video")
        self.model = tk.StringVar(value="btr_best_v2.pt")
        self.source = tk.StringVar(value="tank_real_test.mp4")
        self.conf = tk.StringVar(value="0.15")
        self.imgsz = tk.StringVar(value="640")
        self.stride = tk.StringVar(value="1")
        self.send_width = tk.StringVar(value="960")

        self.build_ui()
        self.root.after(200, self.update_log)

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="UAV AI + ROS2 CONTROL PANEL",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#222222",
            pady=12
        )
        title.pack(fill="x")

        main = tk.Frame(self.root, padx=18, pady=15)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 20))

        right = tk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        tk.Label(left, text="Connection", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))

        tk.Label(left, text="Ubuntu VM IP:").grid(row=1, column=0, sticky="w")
        tk.Entry(left, textvariable=self.vm_ip, width=34).grid(row=2, column=0, sticky="w", pady=(0, 12))

        tk.Label(left, text="Demo Mode", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky="w", pady=(0, 5))

        mode_box = ttk.Combobox(
            left,
            textvariable=self.mode,
            values=list(MODES.keys()),
            width=38,
            state="readonly"
        )
        mode_box.grid(row=4, column=0, sticky="w", pady=(0, 12))
        mode_box.bind("<<ComboboxSelected>>", self.on_mode_change)

        tk.Label(left, text="Model file:").grid(row=5, column=0, sticky="w")
        model_row = tk.Frame(left)
        model_row.grid(row=6, column=0, sticky="w", pady=(0, 10))
        tk.Entry(model_row, textvariable=self.model, width=28).pack(side="left")
        tk.Button(model_row, text="Browse", command=self.browse_model).pack(side="left", padx=5)

        tk.Label(left, text="Video / stream source:").grid(row=7, column=0, sticky="w")
        source_row = tk.Frame(left)
        source_row.grid(row=8, column=0, sticky="w", pady=(0, 10))
        tk.Entry(source_row, textvariable=self.source, width=28).pack(side="left")
        tk.Button(source_row, text="Browse", command=self.browse_source).pack(side="left", padx=5)

        tk.Label(left, text="Detection settings", font=("Arial", 12, "bold")).grid(row=9, column=0, sticky="w", pady=(8, 5))

        settings = tk.Frame(left)
        settings.grid(row=10, column=0, sticky="w")

        tk.Label(settings, text="Confidence:").grid(row=0, column=0, sticky="w")
        tk.Entry(settings, textvariable=self.conf, width=10).grid(row=0, column=1, padx=8, pady=3)

        tk.Label(settings, text="Image size:").grid(row=1, column=0, sticky="w")
        tk.Entry(settings, textvariable=self.imgsz, width=10).grid(row=1, column=1, padx=8, pady=3)

        tk.Label(settings, text="Stride:").grid(row=2, column=0, sticky="w")
        tk.Entry(settings, textvariable=self.stride, width=10).grid(row=2, column=1, padx=8, pady=3)

        tk.Label(settings, text="Send width:").grid(row=3, column=0, sticky="w")
        tk.Entry(settings, textvariable=self.send_width, width=10).grid(row=3, column=1, padx=8, pady=3)

        button_row = tk.Frame(left)
        button_row.grid(row=11, column=0, sticky="w", pady=20)

        self.start_btn = tk.Button(
            button_row,
            text="START AI SENDER",
            width=18,
            bg="#1f8f3a",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.start_sender
        )
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = tk.Button(
            button_row,
            text="STOP",
            width=10,
            bg="#b22222",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.stop_sender,
            state="disabled"
        )
        self.stop_btn.pack(side="left")

        tk.Button(
            left,
            text="Clear Log",
            width=12,
            command=self.clear_log
        ).grid(row=12, column=0, sticky="w")

        status_box = tk.LabelFrame(left, text="Launch order reminder", padx=10, pady=8)
        status_box.grid(row=13, column=0, sticky="we", pady=20)

        reminder = (
            "1. Ubuntu: ~/start_bridge.sh\n"
            "2. Ubuntu: ~/start_dashboard.sh\n"
            "3. Windows: Start AI sender here"
        )
        tk.Label(status_box, text=reminder, justify="left").pack(anchor="w")

        tk.Label(right, text="Sender Log", font=("Arial", 12, "bold")).pack(anchor="w")

        self.log = tk.Text(right, height=30, bg="#0b1020", fg="#e8e8e8", insertbackground="white")
        self.log.pack(fill="both", expand=True)

        self.write_log("Control panel ready.\n")
        self.write_log("Start Ubuntu bridge and dashboard first, then press START AI SENDER.\n\n")

    def on_mode_change(self, event=None):
        selected = self.mode.get()
        data = MODES.get(selected, {})
        self.model.set(data.get("model", ""))
        self.source.set(data.get("source", ""))
        self.conf.set(data.get("conf", "0.25"))

    def browse_model(self):
        file_path = filedialog.askopenfilename(
            initialdir=str(PROJECT_DIR),
            title="Select YOLO model",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.model.set(file_path)

    def browse_source(self):
        file_path = filedialog.askopenfilename(
            initialdir=str(PROJECT_DIR),
            title="Select video/source",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.source.set(file_path)

    def start_sender(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Already running", "The sender is already running.")
            return

        if not SENDER_SCRIPT.exists():
            messagebox.showerror("Missing file", f"Cannot find sender script:\n{SENDER_SCRIPT}")
            return

        model = self.model.get().strip()
        source = self.source.get().strip()

        if not model:
            messagebox.showerror("Missing model", "Please choose a model file.")
            return

        if not source:
            source = simpledialog.askstring("Live stream URL", "Paste live stream URL:")
            if not source:
                return
            self.source.set(source)

        cmd = [
            sys.executable,
            str(SENDER_SCRIPT),
            "--target", self.vm_ip.get().strip(),
            "--source", source,
            "--model", model,
            "--conf", self.conf.get().strip(),
            "--imgsz", self.imgsz.get().strip(),
            "--stride", self.stride.get().strip(),
            "--send_width", self.send_width.get().strip(),
            "--show", "1"
        ]

        self.write_log("\n========================================\n")
        self.write_log("Starting sender...\n")
        self.write_log(f"Mode: {self.mode.get()}\n")
        self.write_log(f"Model: {model}\n")
        self.write_log(f"Source: {source}\n")
        self.write_log(f"Target VM: {self.vm_ip.get().strip()}\n")
        self.write_log("========================================\n\n")

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        except Exception as e:
            messagebox.showerror("Start failed", str(e))
            return

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        threading.Thread(target=self.read_process_output, daemon=True).start()

    def read_process_output(self):
        try:
            for line in self.process.stdout:
                self.log_queue.put(line)
        except Exception as e:
            self.log_queue.put(f"\nLog reader error: {e}\n")
        finally:
            self.log_queue.put("\nSender stopped.\n")

    def stop_sender(self):
        if self.process is not None and self.process.poll() is None:
            self.write_log("\nStopping sender...\n")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def update_log(self):
        while not self.log_queue.empty():
            line = self.log_queue.get()
            self.write_log(line)

        if self.process is not None and self.process.poll() is not None:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

        self.root.after(200, self.update_log)

    def write_log(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def clear_log(self):
        self.log.delete("1.0", "end")


if __name__ == "__main__":
    root = tk.Tk()
    app = UAVAIControlPanel(root)
    root.mainloop()
