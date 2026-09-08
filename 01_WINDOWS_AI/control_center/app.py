"""Polished local GUI for configuring and launching the UAV prototype tools."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import importlib.util
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, TypeVar

from uav_security.model_integrity import ModelIntegrityError, verify_trusted_model
from uav_security.source_urls import source_log_label

from .configuration import (
    DASHBOARD_LAUNCHER,
    LIVE_TESTER_SCRIPT,
    PROJECT_ROOT,
    SOURCE_MODES,
    VIDEO_FILE,
    WINDOWS_AI_ROOT,
    ControlCenterError,
    DashboardSettings,
    LiveTesterSettings,
    MissionSettings,
    SenderLaunchSettings,
    build_dashboard_environment,
    build_live_tester_command,
    build_mission_command,
    build_sender_command,
    build_sender_environment,
    load_local_settings,
    save_local_settings,
)


T = TypeVar("T")

COLORS = {
    "page": "#08111f",
    "panel": "#101c2d",
    "panel_alt": "#16253a",
    "border": "#263b55",
    "text": "#ecf4ff",
    "muted": "#93a7bd",
    "cyan": "#29d3e2",
    "green": "#48d597",
    "amber": "#ffbd59",
    "red": "#ff6b6b",
    "blue": "#367bf5",
}


def _load_dataclass(model: type[T], payload: Any) -> T:
    defaults = asdict(model())
    if isinstance(payload, dict):
        defaults.update({key: value for key, value in payload.items() if key in defaults})
    return model(**defaults)


class UAVPrototypeControlCenter:
    """Single desktop surface for the maintained Windows-side applications."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("UAV AI Prototype Control Center")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 700)
        self.root.configure(bg=COLORS["page"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        saved = load_local_settings()
        self.live_vars = self._make_vars(
            _load_dataclass(LiveTesterSettings, saved.get("live"))
        )
        self.dashboard_vars = self._make_vars(
            _load_dataclass(DashboardSettings, saved.get("dashboard"))
        )
        self.sender_vars = self._make_vars(
            _load_dataclass(SenderLaunchSettings, saved.get("sender"))
        )
        self.mission_vars = self._make_vars(
            _load_dataclass(MissionSettings, saved.get("mission"))
        )
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.process_labels: dict[str, str] = {}
        self.status_text = tk.StringVar(value="Ready")
        self.card_python = tk.StringVar(value="Checking…")
        self.card_gpu = tk.StringVar(value="Checking…")
        self.card_models = tk.StringVar(value="Configure models")
        self.card_processes = tk.StringVar(value="No processes")

        self._configure_style()
        self._build_shell()
        self._write_log("UAV Prototype Control Center ready.\n", "SYSTEM")
        self._write_log(
            "Configure a workflow, review readiness, then launch it from the GUI.\n",
            "SYSTEM",
        )
        self.refresh_readiness()
        self.root.after(100, self._drain_events)

    def _make_vars(self, instance: Any) -> dict[str, tk.Variable]:
        result: dict[str, tk.Variable] = {}
        for field in fields(instance):
            value = getattr(instance, field.name)
            result[field.name] = (
                tk.BooleanVar(value=value)
                if isinstance(value, bool)
                else tk.StringVar(value=str(value))
            )
        return result

    def _settings(self, model: type[T], variables: dict[str, tk.Variable]) -> T:
        return model(**{field.name: variables[field.name].get() for field in fields(model)})

    def _configure_style(self) -> None:
        self.root.option_add("*Font", ("Segoe UI", 10))
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["page"])
        style.configure(
            "Panel.TFrame", background=COLORS["panel"], relief="flat"
        )
        style.configure(
            "TLabel", background=COLORS["page"], foreground=COLORS["text"]
        )
        style.configure(
            "Muted.TLabel", background=COLORS["page"], foreground=COLORS["muted"]
        )
        style.configure(
            "Header.TLabel",
            background=COLORS["page"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 22),
        )
        style.configure(
            "CardTitle.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["muted"],
            font=("Segoe UI Semibold", 9),
        )
        style.configure(
            "CardValue.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 13),
        )
        style.configure(
            "TLabelframe",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
        )
        style.configure(
            "TLabelframe.Label",
            background=COLORS["panel"],
            foreground=COLORS["cyan"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "TEntry",
            fieldbackground=COLORS["panel_alt"],
            foreground=COLORS["text"],
            insertcolor=COLORS["text"],
            bordercolor=COLORS["border"],
        )
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["panel_alt"],
            background=COLORS["panel_alt"],
            foreground=COLORS["text"],
            arrowcolor=COLORS["cyan"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", COLORS["panel_alt"])],
            foreground=[("readonly", COLORS["text"])],
        )
        style.configure(
            "TCheckbutton", background=COLORS["panel"], foreground=COLORS["text"]
        )
        style.map("TCheckbutton", background=[("active", COLORS["panel"])])
        style.configure(
            "Accent.TButton",
            background=COLORS["blue"],
            foreground="white",
            padding=(14, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Accent.TButton", background=[("active", "#4d8cff")])
        style.configure(
            "Secondary.TButton",
            background=COLORS["panel_alt"],
            foreground=COLORS["text"],
            padding=(12, 7),
        )
        style.map("Secondary.TButton", background=[("active", COLORS["border"])])
        style.configure(
            "Danger.TButton",
            background="#7f2d3a",
            foreground="white",
            padding=(12, 7),
        )
        style.configure("TNotebook", background=COLORS["page"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=COLORS["panel"],
            foreground=COLORS["muted"],
            padding=(18, 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["panel_alt"])],
            foreground=[("selected", COLORS["cyan"])],
        )
        style.configure(
            "Treeview",
            background=COLORS["panel"],
            fieldbackground=COLORS["panel"],
            foreground=COLORS["text"],
            rowheight=28,
            bordercolor=COLORS["border"],
        )
        style.configure(
            "Treeview.Heading",
            background=COLORS["panel_alt"],
            foreground=COLORS["cyan"],
            font=("Segoe UI Semibold", 9),
        )

    def _build_shell(self) -> None:
        header = ttk.Frame(self.root, padding=(22, 16, 22, 10))
        header.pack(fill="x")
        ttk.Label(header, text="UAV AI Prototype", style="Header.TLabel").pack(
            side="left"
        )
        ttk.Label(
            header,
            text="CONTROL CENTER  •  LOCAL / ADVISORY",
            foreground=COLORS["cyan"],
            font=("Segoe UI Semibold", 9),
        ).pack(side="left", padx=18, pady=(8, 0))
        ttk.Button(
            header,
            text="Stop all",
            style="Danger.TButton",
            command=self.stop_all,
        ).pack(side="right")
        ttk.Button(
            header,
            text="Save settings",
            style="Secondary.TButton",
            command=self.save_settings,
        ).pack(side="right", padx=8)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        self.dashboard_tab = ttk.Frame(self.notebook, padding=14)
        self.live_tab = ttk.Frame(self.notebook, padding=14)
        self.sender_tab = ttk.Frame(self.notebook, padding=14)
        self.mission_tab = ttk.Frame(self.notebook, padding=14)
        self.log_tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.live_tab, text="Live Tester")
        self.notebook.add(self.sender_tab, text="Secure Sender")
        self.notebook.add(self.mission_tab, text="Mission Copilot")
        self.notebook.add(self.log_tab, text="Activity Log")

        self._build_dashboard()
        self._build_live_tab()
        self._build_sender_tab()
        self._build_mission_tab()
        self._build_log_tab()

        footer = ttk.Frame(self.root, padding=(20, 5, 20, 10))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status_text, style="Muted.TLabel").pack(
            side="left"
        )
        ttk.Label(
            footer,
            text="No model weights, media, credentials, or generated outputs are committed",
            style="Muted.TLabel",
        ).pack(side="right")

    def _build_dashboard(self) -> None:
        tab = self.dashboard_tab
        tab.columnconfigure(0, weight=1)
        ttk.Label(
            tab,
            text="System dashboard",
            font=("Segoe UI Semibold", 17),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            tab,
            text="Readiness, trusted artifacts, applications and local process state.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 14))

        cards = ttk.Frame(tab)
        cards.grid(row=2, column=0, sticky="ew")
        for column in range(4):
            cards.columnconfigure(column, weight=1)
        self._card(cards, 0, "PYTHON", self.card_python, COLORS["cyan"])
        self._card(cards, 1, "COMPUTE", self.card_gpu, COLORS["green"])
        self._card(cards, 2, "MODELS", self.card_models, COLORS["amber"])
        self._card(cards, 3, "PROCESSES", self.card_processes, COLORS["blue"])

        body = ttk.Frame(tab)
        body.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        tab.rowconfigure(3, weight=1)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        readiness = ttk.LabelFrame(body, text="Readiness", padding=10)
        readiness.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        readiness.rowconfigure(0, weight=1)
        readiness.columnconfigure(0, weight=1)
        self.readiness_tree = ttk.Treeview(
            readiness,
            columns=("status", "detail"),
            show="headings",
            selectmode="browse",
        )
        self.readiness_tree.heading("status", text="Status")
        self.readiness_tree.heading("detail", text="Component / details")
        self.readiness_tree.column("status", width=105, anchor="center")
        self.readiness_tree.column("detail", width=600, anchor="w")
        self.readiness_tree.grid(row=0, column=0, sticky="nsew")
        ttk.Button(
            readiness,
            text="Refresh readiness",
            style="Secondary.TButton",
            command=self.refresh_readiness,
        ).grid(row=1, column=0, sticky="w", pady=(10, 0))

        actions = ttk.LabelFrame(body, text="Quick actions", padding=14)
        actions.grid(row=0, column=1, sticky="nsew")
        actions.columnconfigure(0, weight=1)
        ttk.Label(actions, text="Recorded dashboard port", background=COLORS["panel"]).grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        ttk.Entry(actions, textvariable=self.dashboard_vars["port"], width=12).grid(
            row=1, column=0, sticky="ew", pady=(0, 10)
        )
        quick = (
            ("Launch configured live tester", self.launch_live, "Accent.TButton"),
            ("Open recorded-analysis dashboard", self.launch_recorded_dashboard, "Secondary.TButton"),
            ("Start secure Windows sender", self.launch_sender, "Secondary.TButton"),
            ("Run Mission Copilot", self.launch_mission, "Secondary.TButton"),
            ("Open local output folder", self.open_output_folder, "Secondary.TButton"),
            ("Open operator documentation", self.open_documentation, "Secondary.TButton"),
        )
        for row, (label, command, style) in enumerate(quick):
            ttk.Button(actions, text=label, style=style, command=command).grid(
                row=row + 2, column=0, sticky="ew", pady=5
            )
        ttk.Label(
            actions,
            text=(
                "Live V2 remains fail-closed: the selected external checkpoint must "
                "be hash-approved and match the six-layer MC Dropout architecture."
            ),
            style="Muted.TLabel",
            wraplength=320,
            justify="left",
        ).grid(row=len(quick) + 2, column=0, sticky="ew", pady=(18, 0))

    def _card(
        self,
        parent: ttk.Frame,
        column: int,
        title: str,
        variable: tk.StringVar,
        accent: str,
    ) -> None:
        frame = tk.Frame(
            parent,
            bg=COLORS["panel"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            padx=14,
            pady=12,
        )
        frame.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 6))
        tk.Label(
            frame,
            text=title,
            bg=COLORS["panel"],
            fg=accent,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w")
        tk.Label(
            frame,
            textvariable=variable,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w", pady=(5, 0))

    def _build_live_tab(self) -> None:
        tab = self.live_tab
        for column in range(3):
            tab.columnconfigure(column, weight=1)
        ttk.Label(tab, text="Live perception and uncertainty", font=("Segoe UI Semibold", 17)).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            tab,
            text="Configure every maintained live tester option. V1 and V2 run only on demand after U freezes the exact frame.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 14))

        models = ttk.LabelFrame(tab, text="Models and source", padding=12)
        models.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        self._entry(models, 0, "Base detector (.pt)", self.live_vars["model_path"], lambda: self._browse_file(self.live_vars["model_path"], "Select trusted base detector", (("PyTorch checkpoint", "*.pt"),)))
        self._entry(models, 1, "V2 MC Dropout (.pt)", self.live_vars["mcdo_model_path"], lambda: self._browse_file(self.live_vars["mcdo_model_path"], "Select validated V2 checkpoint", (("PyTorch checkpoint", "*.pt"),)))
        self._combo(models, 2, "Source mode", self.live_vars["source_mode"], SOURCE_MODES)
        self._entry(models, 3, "MP4 file", self.live_vars["video_path"], lambda: self._browse_file(self.live_vars["video_path"], "Select MP4", (("MP4 video", "*.mp4"),)))
        self._check(models, 4, "Loop video", self.live_vars["loop_video"])
        self._entry(models, 5, "Monitor", self.live_vars["monitor"], width=12)
        region = ttk.Frame(models, style="Panel.TFrame")
        region.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for column, key in enumerate(("left", "top", "width", "height")):
            ttk.Label(region, text=key.title(), background=COLORS["panel"]).grid(row=0, column=column, sticky="w", padx=3)
            ttk.Entry(region, textvariable=self.live_vars[key], width=8).grid(row=1, column=column, padx=3)

        inference = ttk.LabelFrame(tab, text="Inference", padding=12)
        inference.grid(row=2, column=1, sticky="nsew", padx=7)
        self._entry(inference, 0, "Confidence", self.live_vars["confidence"], width=14)
        self._entry(inference, 1, "NMS IoU", self.live_vars["iou"], width=14)
        self._entry(inference, 2, "Image size", self.live_vars["image_size"], width=14)
        self._combo(inference, 3, "Device", self.live_vars["device"], ("auto", "cpu", "cuda", "cuda:0"))
        self._entry(inference, 4, "Maximum FPS (0 = uncapped)", self.live_vars["max_fps"], width=14)
        self._entry(inference, 5, "Classes", self.live_vars["classes"])
        self._check(inference, 6, "Tank-only display", self.live_vars["tank_only"])
        self._entry(inference, 7, "Output directory", self.live_vars["output_directory"], lambda: self._browse_directory(self.live_vars["output_directory"], "Select output directory"))
        self._entry(inference, 8, "Trusted-model registry", self.live_vars["registry_path"], lambda: self._browse_file(self.live_vars["registry_path"], "Select trusted-model registry", (("CSV registry", "*.csv"),)))

        uncertainty = ttk.LabelFrame(tab, text="Uncertainty inspection", padding=12)
        uncertainty.grid(row=2, column=2, sticky="nsew", padx=(7, 0))
        self._check(uncertainty, 0, "Enable U-key inspection", self.live_vars["uncertainty_enabled"])
        self._entry(uncertainty, 1, "V1 perturbation samples", self.live_vars["uncertainty_samples"], width=14)
        self._entry(uncertainty, 2, "V1 deterministic seed", self.live_vars["uncertainty_seed"], width=14)
        self._entry(uncertainty, 3, "V1 matching IoU", self.live_vars["uncertainty_match_iou"], width=14)
        ttk.Separator(uncertainty).grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)
        self._check(uncertainty, 5, "Enable V2 MC Dropout", self.live_vars["mcdo_enabled"])
        self._check(uncertainty, 6, "Require V2 validation", self.live_vars["mcdo_required"])
        self._entry(uncertainty, 7, "V2 stochastic passes", self.live_vars["mcdo_passes"], width=14)
        self._entry(uncertainty, 8, "V2 matching IoU", self.live_vars["mcdo_match_iou"], width=14)
        ttk.Label(
            uncertainty,
            text="V2 uses one unchanged frame, BatchNorm eval, and only the six validated dropout modules in train mode.",
            style="Muted.TLabel",
            wraplength=300,
        ).grid(row=18, column=0, columnspan=3, sticky="w", pady=(14, 0))

        actions = ttk.Frame(tab, padding=(0, 14, 0, 0))
        actions.grid(row=3, column=0, columnspan=3, sticky="ew")
        ttk.Button(actions, text="Launch live tester", style="Accent.TButton", command=self.launch_live).pack(side="left")
        ttk.Button(actions, text="Stop live tester", style="Danger.TButton", command=lambda: self.stop_process("live")).pack(side="left", padx=8)
        ttk.Button(actions, text="Save settings", style="Secondary.TButton", command=self.save_settings).pack(side="left")
        ttk.Button(actions, text="Open activity log", style="Secondary.TButton", command=lambda: self.notebook.select(self.log_tab)).pack(side="right")

    def _build_sender_tab(self) -> None:
        tab = self.sender_tab
        for column in range(3):
            tab.columnconfigure(column, weight=1)
        ttk.Label(tab, text="Secure Windows → ROS 2 sender", font=("Segoe UI Semibold", 17)).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(tab, text="Existing YOLO + BoT-SORT sender with explicit mutual-TLS and transport configuration.", style="Muted.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 14))

        source = ttk.LabelFrame(tab, text="Source and model", padding=12)
        source.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        self._entry(source, 0, "Model (.pt)", self.sender_vars["model_path"], lambda: self._browse_file(self.sender_vars["model_path"], "Select sender model", (("PyTorch checkpoint", "*.pt"),)))
        self._entry(source, 1, "Video / stream source", self.sender_vars["source"], lambda: self._browse_file(self.sender_vars["source"], "Select sender video", (("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))))
        self._entry(source, 2, "Bridge address", self.sender_vars["target"])
        self._entry(source, 3, "Bridge port", self.sender_vars["port"], width=14)
        self._entry(source, 4, "TLS server name", self.sender_vars["tls_server_name"])

        tls = ttk.LabelFrame(tab, text="Mutual TLS", padding=12)
        tls.grid(row=2, column=1, sticky="nsew", padx=7)
        self._entry(tls, 0, "Sender certificate", self.sender_vars["tls_certificate"], lambda: self._browse_file(self.sender_vars["tls_certificate"], "Select sender certificate", (("Certificate", "*.crt *.pem"), ("All files", "*.*"))))
        self._entry(tls, 1, "Sender private key", self.sender_vars["tls_private_key"], lambda: self._browse_file(self.sender_vars["tls_private_key"], "Select sender private key", (("Private key", "*.key *.pem"), ("All files", "*.*"))))
        self._entry(tls, 2, "Bridge CA certificate", self.sender_vars["tls_ca_certificate"], lambda: self._browse_file(self.sender_vars["tls_ca_certificate"], "Select bridge CA", (("CA certificate", "*.crt *.pem"), ("All files", "*.*"))))
        ttk.Label(tls, text="Credential contents are never displayed or logged. Local paths are stored only in the ignored settings file.", style="Muted.TLabel", wraplength=300).grid(row=7, column=0, columnspan=3, sticky="w", pady=(16, 0))

        inference = ttk.LabelFrame(tab, text="Sender inference", padding=12)
        inference.grid(row=2, column=2, sticky="nsew", padx=(7, 0))
        self._entry(inference, 0, "Confidence", self.sender_vars["confidence"], width=14)
        self._entry(inference, 1, "NMS IoU", self.sender_vars["iou"], width=14)
        self._entry(inference, 2, "Image size", self.sender_vars["image_size"], width=14)
        self._entry(inference, 3, "Frame stride", self.sender_vars["stride"], width=14)
        self._entry(inference, 4, "Send width", self.sender_vars["send_width"], width=14)
        self._entry(inference, 5, "Tracker", self.sender_vars["tracker"])
        self._check(inference, 6, "Show local preview", self.sender_vars["show"])
        self._check(inference, 7, "Military classes only", self.sender_vars["military_only"])

        actions = ttk.Frame(tab, padding=(0, 14, 0, 0))
        actions.grid(row=3, column=0, columnspan=3, sticky="ew")
        ttk.Button(actions, text="Start secure sender", style="Accent.TButton", command=self.launch_sender).pack(side="left")
        ttk.Button(actions, text="Stop sender", style="Danger.TButton", command=lambda: self.stop_process("sender")).pack(side="left", padx=8)
        ttk.Button(actions, text="Open sender documentation", style="Secondary.TButton", command=self.open_documentation).pack(side="left")

    def _build_mission_tab(self) -> None:
        tab = self.mission_tab
        tab.columnconfigure(0, weight=1)
        ttk.Label(tab, text="Deterministic Mission Copilot", font=("Segoe UI Semibold", 17)).grid(row=0, column=0, sticky="w")
        ttk.Label(tab, text="Offline advisory planning only—no PX4, actuator, velocity or flight-control outputs.", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 14))
        panel = ttk.LabelFrame(tab, text="Mission inputs", padding=16)
        panel.grid(row=2, column=0, sticky="nsew")
        panel.columnconfigure(1, weight=1)
        self._entry(panel, 0, "Scenario JSON", self.mission_vars["scenario_path"], lambda: self._browse_file(self.mission_vars["scenario_path"], "Select mission scenario", (("JSON", "*.json"),)))
        self._entry(panel, 1, "Safety policy JSON", self.mission_vars["policy_path"], lambda: self._browse_file(self.mission_vars["policy_path"], "Select mission policy", (("JSON", "*.json"),)))
        self._entry(panel, 2, "Output JSON", self.mission_vars["output_path"], lambda: self._browse_save(self.mission_vars["output_path"], "Select mission output"))
        self._check(panel, 3, "Verbose explanation", self.mission_vars["verbose"])
        self._check(panel, 4, "Validate scenario only", self.mission_vars["validate_only"])
        self._check(panel, 5, "Require every task to be assigned", self.mission_vars["require_complete"])
        actions = ttk.Frame(tab, padding=(0, 14, 0, 0))
        actions.grid(row=3, column=0, sticky="ew")
        ttk.Button(actions, text="Run Mission Copilot", style="Accent.TButton", command=self.launch_mission).pack(side="left")
        ttk.Button(actions, text="Stop mission process", style="Danger.TButton", command=lambda: self.stop_process("mission")).pack(side="left", padx=8)
        ttk.Button(actions, text="Open output folder", style="Secondary.TButton", command=self.open_output_folder).pack(side="left")

    def _build_log_tab(self) -> None:
        self.log_tab.rowconfigure(1, weight=1)
        self.log_tab.columnconfigure(0, weight=1)
        top = ttk.Frame(self.log_tab)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Application activity", font=("Segoe UI Semibold", 17)).pack(side="left")
        ttk.Button(top, text="Clear", style="Secondary.TButton", command=lambda: self.log.delete("1.0", "end")).pack(side="right")
        self.log = tk.Text(
            self.log_tab,
            bg="#050b14",
            fg="#d7e6f7",
            insertbackground="white",
            relief="flat",
            padx=12,
            pady=10,
            font=("Cascadia Mono", 9),
            wrap="word",
        )
        self.log.grid(row=1, column=0, sticky="nsew")
        self.log.tag_configure("SYSTEM", foreground=COLORS["cyan"])
        self.log.tag_configure("ERROR", foreground=COLORS["red"])
        self.log.tag_configure("PROCESS", foreground=COLORS["green"])

    def _entry(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        variable: tk.Variable,
        browse: Any | None = None,
        width: int = 34,
    ) -> None:
        grid_row = row * 2
        ttk.Label(parent, text=label, background=COLORS["panel"]).grid(
            row=grid_row, column=0, columnspan=3, sticky="w", pady=(5, 2)
        )
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=grid_row + 1, column=0, columnspan=2, sticky="ew", pady=(0, 5)
        )
        parent.columnconfigure(0, weight=1)
        if browse is not None:
            ttk.Button(parent, text="Browse", style="Secondary.TButton", command=browse).grid(
                row=grid_row + 1, column=2, padx=(6, 0), pady=(0, 5)
            )

    def _combo(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        variable: tk.Variable,
        values: tuple[str, ...],
    ) -> None:
        grid_row = row * 2
        ttk.Label(parent, text=label, background=COLORS["panel"]).grid(
            row=grid_row, column=0, columnspan=3, sticky="w", pady=(5, 2)
        )
        ttk.Combobox(
            parent, textvariable=variable, values=values, state="readonly"
        ).grid(row=grid_row + 1, column=0, columnspan=3, sticky="ew", pady=(0, 5))

    def _check(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        variable: tk.Variable,
    ) -> None:
        ttk.Checkbutton(parent, text=label, variable=variable).grid(
            row=row * 2, column=0, columnspan=3, sticky="w", pady=5
        )

    def _browse_file(
        self,
        variable: tk.Variable,
        title: str,
        filetypes: tuple[tuple[str, str], ...],
    ) -> None:
        current = Path(str(variable.get())).expanduser()
        initial = current.parent if current.parent.is_dir() else PROJECT_ROOT
        selected = filedialog.askopenfilename(
            parent=self.root,
            title=title,
            initialdir=str(initial),
            filetypes=filetypes,
        )
        if selected:
            variable.set(selected)
            self.refresh_readiness()

    def _browse_directory(self, variable: tk.Variable, title: str) -> None:
        selected = filedialog.askdirectory(parent=self.root, title=title)
        if selected:
            variable.set(selected)

    def _browse_save(self, variable: tk.Variable, title: str) -> None:
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title=title,
            defaultextension=".json",
            filetypes=(("JSON", "*.json"),),
        )
        if selected:
            variable.set(selected)

    def current_live_settings(self) -> LiveTesterSettings:
        return self._settings(LiveTesterSettings, self.live_vars)

    def current_sender_settings(self) -> SenderLaunchSettings:
        return self._settings(SenderLaunchSettings, self.sender_vars)

    def current_dashboard_settings(self) -> DashboardSettings:
        return self._settings(DashboardSettings, self.dashboard_vars)

    def current_mission_settings(self) -> MissionSettings:
        return self._settings(MissionSettings, self.mission_vars)

    def save_settings(self, *, notify: bool = True) -> None:
        try:
            path = save_local_settings(
                live=self.current_live_settings(),
                dashboard=self.current_dashboard_settings(),
                sender=self.current_sender_settings(),
                mission=self.current_mission_settings(),
            )
        except OSError as error:
            if notify:
                messagebox.showerror("Save failed", str(error), parent=self.root)
            return
        self.status_text.set(f"Settings saved locally: {path.name}")
        if notify:
            self._write_log(f"Settings saved under ignored output storage: {path}\n", "SYSTEM")

    def _verify_model(self, model_path: str, registry_path: str) -> str:
        registry = Path(registry_path).expanduser() if registry_path.strip() else None
        digest = verify_trusted_model(model_path, registry)
        return digest[:12]

    def refresh_readiness(self) -> None:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.card_python.set(f"Python {python_version}")
        try:
            import torch

            if torch.cuda.is_available():
                self.card_gpu.set(torch.cuda.get_device_name(0)[:24])
            else:
                self.card_gpu.set("CPU mode")
        except Exception:
            self.card_gpu.set("Torch unavailable")

        live = self.current_live_settings()
        model_count = sum(
            bool(value.strip()) for value in (live.model_path, live.mcdo_model_path)
        )
        self.card_models.set(f"{model_count}/2 selected")
        active = [
            self.process_labels[key]
            for key, process in self.processes.items()
            if process.poll() is None
        ]
        self.card_processes.set(", ".join(active) if active else "No processes")

        rows: list[tuple[str, str]] = []
        rows.append(("READY" if Path(sys.executable).is_file() else "ERROR", f"Python — {sys.executable}"))
        rows.append(("READY" if LIVE_TESTER_SCRIPT.is_file() else "ERROR", "Live tester entry point"))
        rows.append(("READY" if DASHBOARD_LAUNCHER.is_file() else "ERROR", "Recorded-analysis dashboard"))
        dashboard_modules = ("gradio", "cv2", "ultralytics", "torch")
        missing = [name for name in dashboard_modules if importlib.util.find_spec(name) is None]
        rows.append(("READY" if not missing else "SETUP", "Dashboard dependencies" + ("" if not missing else f" — missing {', '.join(missing)}")))
        try:
            dashboard_environment = build_dashboard_environment(
                self.current_dashboard_settings()
            )
            rows.append(
                (
                    "READY",
                    f"Recorded dashboard — http://127.0.0.1:{dashboard_environment['UAV_DASHBOARD_PORT']}",
                )
            )
        except ControlCenterError as error:
            rows.append(("BLOCKED", str(error)))
        for label, path in (("Base detector", live.model_path), ("V2 checkpoint", live.mcdo_model_path)):
            if not path.strip():
                rows.append(("SETUP", f"{label} — not selected"))
                continue
            try:
                digest = self._verify_model(path, live.registry_path)
                rows.append(("TRUSTED", f"{label} — SHA-256 {digest}…"))
            except (ModelIntegrityError, OSError) as error:
                rows.append(("BLOCKED", f"{label} — {error}"))
        if live.source_mode == VIDEO_FILE:
            ready = Path(live.video_path).expanduser().is_file()
            rows.append(("READY" if ready else "SETUP", "Video file source"))
        else:
            rows.append(("READY", f"Source mode — {live.source_mode}"))
        mission = Path(str(self.mission_vars["scenario_path"].get())).expanduser()
        rows.append(("READY" if mission.is_file() else "SETUP", "Mission Copilot scenario"))

        self.readiness_tree.delete(*self.readiness_tree.get_children())
        for status, detail in rows:
            self.readiness_tree.insert("", "end", values=(status, detail))
        self.status_text.set("Readiness refreshed")

    def launch_live(self) -> None:
        settings = self.current_live_settings()
        try:
            self._verify_model(settings.model_path, settings.registry_path)
            if settings.mcdo_enabled:
                self._verify_model(settings.mcdo_model_path, settings.registry_path)
            command = build_live_tester_command(settings, sys.executable)
        except (ControlCenterError, ModelIntegrityError) as error:
            messagebox.showerror("Live tester not ready", str(error), parent=self.root)
            return
        self.save_settings(notify=False)
        self._write_log(
            f"Launching live tester: {settings.source_mode}; base={Path(settings.model_path).name}; V2={'enabled' if settings.mcdo_enabled else 'disabled'}\n",
            "SYSTEM",
        )
        self._start_process("live", "Live tester", command)

    def launch_recorded_dashboard(self) -> None:
        dashboard_source = WINDOWS_AI_ROOT / "model_test_dashboard" / "src"
        required = ("gradio", "cv2", "ultralytics", "torch")
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            messagebox.showerror(
                "Dashboard dependencies missing",
                "Install the dashboard requirements in the selected environment: "
                + ", ".join(missing),
                parent=self.root,
            )
            return
        command = [sys.executable, "-u", "-m", "uav_model_dashboard.app"]
        extra = {
            "PYTHONPATH": os.pathsep.join((str(dashboard_source), str(PROJECT_ROOT))),
            "UAV_YOLO_PYTHON": sys.executable,
        }
        try:
            extra.update(build_dashboard_environment(self.current_dashboard_settings()))
        except ControlCenterError as error:
            messagebox.showerror("Dashboard not ready", str(error), parent=self.root)
            return
        initial_model = str(self.live_vars["model_path"].get()).strip()
        if initial_model:
            extra["UAV_MODEL_PATH"] = initial_model
        self._start_process("dashboard", "Recorded dashboard", command, extra)

    def launch_sender(self) -> None:
        settings = self.current_sender_settings()
        try:
            self._verify_model(
                settings.model_path, str(self.live_vars["registry_path"].get())
            )
            command = build_sender_command(settings, sys.executable)
            environment = build_sender_environment(settings)
        except (ControlCenterError, ModelIntegrityError) as error:
            messagebox.showerror("Sender not ready", str(error), parent=self.root)
            return
        self.save_settings(notify=False)
        self._write_log(
            f"Launching secure sender to {settings.target}:{settings.port}; source={source_log_label(settings.source)}\n",
            "SYSTEM",
        )
        self._start_process("sender", "Secure sender", command, environment)

    def launch_mission(self) -> None:
        settings = self.current_mission_settings()
        try:
            command, environment = build_mission_command(settings, sys.executable)
        except ControlCenterError as error:
            messagebox.showerror("Mission Copilot not ready", str(error), parent=self.root)
            return
        self.save_settings(notify=False)
        self._write_log(
            f"Running Mission Copilot with {Path(settings.scenario_path).name}\n",
            "SYSTEM",
        )
        self._start_process("mission", "Mission Copilot", command, environment)

    def _start_process(
        self,
        key: str,
        label: str,
        command: list[str],
        extra_environment: dict[str, str] | None = None,
    ) -> None:
        existing = self.processes.get(key)
        if existing is not None and existing.poll() is None:
            messagebox.showwarning("Already running", f"{label} is already running.", parent=self.root)
            return
        environment = os.environ.copy()
        environment.update(extra_environment or {})
        environment["PYTHONUNBUFFERED"] = "1"
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
        except OSError as error:
            messagebox.showerror("Launch failed", str(error), parent=self.root)
            return
        self.processes[key] = process
        self.process_labels[key] = label
        self.status_text.set(f"{label} running")
        self._write_log(f"{label} started (PID {process.pid}).\n", "PROCESS")
        threading.Thread(
            target=self._read_process,
            args=(key, label, process),
            daemon=True,
        ).start()
        self.refresh_readiness()

    def _read_process(
        self,
        key: str,
        label: str,
        process: subprocess.Popen[str],
    ) -> None:
        stream = process.stdout
        if stream is not None:
            for line in stream:
                self.events.put(("PROCESS", f"[{label}] {line}"))
        return_code = process.wait()
        self.events.put(("EXIT", f"{key}|{label}|{return_code}"))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "EXIT":
                    key, label, return_code = payload.split("|", 2)
                    self._write_log(
                        f"{label} stopped with exit code {return_code}.\n",
                        "PROCESS" if return_code == "0" else "ERROR",
                    )
                    process = self.processes.get(key)
                    if process is not None and process.poll() is not None:
                        self.processes.pop(key, None)
                        self.process_labels.pop(key, None)
                    self.status_text.set(f"{label} stopped")
                    self.refresh_readiness()
                else:
                    self._write_log(payload, kind)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _write_log(self, text: str, tag: str = "PROCESS") -> None:
        if not hasattr(self, "log"):
            return
        self.log.insert("end", text, tag)
        self.log.see("end")

    def stop_process(self, key: str) -> None:
        process = self.processes.get(key)
        label = self.process_labels.get(key, key)
        if process is None or process.poll() is not None:
            self.status_text.set(f"{label} is not running")
            return
        self._write_log(f"Stopping {label}…\n", "SYSTEM")
        process.terminate()

    def stop_all(self) -> None:
        for key in tuple(self.processes):
            self.stop_process(key)

    def open_output_folder(self) -> None:
        destination = PROJECT_ROOT / "08_OUTPUTS"
        destination.mkdir(parents=True, exist_ok=True)
        self._open_path(destination)

    def open_documentation(self) -> None:
        self._open_path(PROJECT_ROOT / "07_DOCUMENTATION")

    def _open_path(self, path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                raise OSError("Folder opening is currently supported on Windows only")
        except OSError as error:
            messagebox.showerror("Could not open path", str(error), parent=self.root)

    def close(self) -> None:
        self.save_settings(notify=False)
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    """Launch the desktop control center or perform a non-interactive UI smoke test."""

    parser = argparse.ArgumentParser(description="UAV AI Prototype Control Center")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = tk.Tk()
    if args.smoke_test:
        root.withdraw()
    app = UAVPrototypeControlCenter(root)
    if args.smoke_test:
        root.update_idletasks()
        print(f"Control center UI smoke test passed: {len(app.notebook.tabs())} tabs")
        root.destroy()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
