"""Temporary, opt-in input-pipeline/GPU diagnostics; no training hooks in the model."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import torch
from pytorch_lightning import Callback
from pytorch_lightning.profilers import PyTorchProfiler


class GPUActivityMonitor(Callback):
    """Sample all local GPUs at 1 Hz for the short profiling window only."""

    def __init__(self, directory, batches=40):
        self.directory = Path(directory)
        self.batches = batches
        self._process = None
        self._stream = None
        self._seen = 0

    def on_train_start(self, trainer, pl_module):
        executable = shutil.which('nvidia-smi')
        if trainer.local_rank != 0 or not torch.cuda.is_available() or executable is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f'gpu-activity-{socket.gethostname()}-{os.getpid()}.csv'
        self._stream = path.open('w')
        try:
            self._process = subprocess.Popen([
                executable, '--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total',
                '--format=csv', '--loop-ms=1000',
            ], stdout=self._stream, stderr=subprocess.STDOUT)
        except OSError:
            self.close()
            return
        print(f'[profile] GPU activity: {path}', flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._seen += 1
        if self._seen >= self.batches:
            self.close()

    def close(self):
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def on_train_end(self, trainer, pl_module):
        self.close()

    def on_exception(self, trainer, pl_module, exception):
        self.close()

    def teardown(self, trainer, pl_module, stage):
        self.close()


class TraceWriter:
    def __init__(self, directory):
        self.directory = Path(directory)

    def __call__(self, profiler):
        self.directory.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        stem = self.directory / f'rank{rank}-{socket.gethostname()}-{os.getpid()}-{time.time_ns()}'
        profiler.export_chrome_trace(str(stem) + '.trace.json')
        events = profiler.key_averages()
        stages = {}
        for name in ('train_dataloader_next', 'batch_to_device', 'training_step', 'backward', 'optimizer_step'):
            matches = [event for event in events if name in event.key and '[pl]' in event.key]
            stages[name] = [dict(
                name=e.key, calls=e.count, cpu_total_ms=e.cpu_time_total / 1000,
                cpu_mean_ms=e.cpu_time_total / max(e.count, 1) / 1000,
                device_total_ms=getattr(e, 'device_time_total', 0) / 1000,
            ) for e in matches]
        (Path(str(stem) + '.summary.json')).write_text(json.dumps(stages, indent=2))
        report = [
            'Timing window: 20 active training batches after 10 skipped + 2 warm-up batches.',
            'Dataloader-next measures exposed main-process waiting, not all worker preprocessing.',
            'Stage totals are nested: optimizer_step includes the forward/backward closure.',
            'Inspect the trace GPU lanes for idle gaps and correlate them with dataloader-next/transfer.',
            'Profiler overhead affects timings; GPU activity CSV is an independent 1 Hz sample.',
            '', json.dumps(stages, indent=2), '', 'CPU operators (self time):',
            events.table(sort_by='self_cpu_time_total', row_limit=30),
        ]
        if torch.cuda.is_available():
            report += ['', 'GPU operators (self time):',
                       events.table(sort_by='self_cuda_time_total', row_limit=30)]
        Path(str(stem) + '.txt').write_text('\n'.join(report))
        print(f'[profile rank {rank}] Timing report: {stem}.txt; GPU timeline: {stem}.trace.json', flush=True)


def training_diagnostics(directory):
    """Capture once, then stop recording while normal training continues."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = PyTorchProfiler(
        dirpath=str(directory), filename='training-profile',
        activities=activities,
        schedule=torch.profiler.schedule(wait=10, warmup=2, active=20, repeat=1),
        on_trace_ready=TraceWriter(directory),
        record_shapes=False, profile_memory=False, with_stack=False,
        record_module_names=False,
    )
    return profiler, GPUActivityMonitor(directory)
