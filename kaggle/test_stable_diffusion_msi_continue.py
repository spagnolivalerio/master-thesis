"""Verifica la continuazione SD senza Kaggle, download o pesi preaddestrati."""

import ast
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import wasserstein_distance
import tifffile
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from tqdm import tqdm


NOTEBOOK = Path(__file__).with_name("stable-diffusion-msi-continue-sd.ipynb")


def notebook_functions():
    namespace = {
        "Path": Path, "json": json, "torch": torch, "np": np, "F": F,
        "Counter": Counter, "math": math, "random": random, "shutil": shutil,
        "hashlib": hashlib, "contextmanager": contextmanager,
        "datetime": datetime, "timezone": timezone, "perf_counter": perf_counter,
        "Dataset": Dataset, "Sampler": Sampler,
        "diffusers": SimpleNamespace(__version__="test"),
        "csv": csv, "plt": plt, "Image": Image,
        "wasserstein_distance": wasserstein_distance, "tifffile": tifffile,
        "tqdm": lambda *args, **kwargs: tqdm(*args, disable=True, **kwargs),
    }
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or source.lstrip().startswith("%"):
            continue
        tree = ast.parse(source, filename=cell["id"])
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), cell["id"], "exec"), namespace)
    return namespace


def original_config():
    notebook = json.loads(NOTEBOOK.with_name("stable-diffusion-msi.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "config")
    assignment = next(
        node for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "config" for target in node.targets)
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "source_config", "exec"), namespace)
    return namespace["config"]


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(0.2), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.tensor(0.3))
        self.config = {"in_channels": 1, "out_channels": 1}

    def forward(self, noisy, timestep, encoder_hidden_states):
        return SimpleNamespace(sample=noisy * (self.base + self.adapter))


class ContinuationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input"
        self.working = self.root / "working"
        self.ns = notebook_functions()
        self.config = original_config()
        self.name = self.config["experiment"]["name"]
        base = self.input / "previous" / "experiments" / self.name
        self.sd_path = base / "sd_checkpoints" / f"{self.name}_sd-run_2" / "latest.pt"
        self.vae_path = base / "vae_checkpoints" / f"{self.name}_vae-run_1" / "best.pt"
        self.sd_path.parent.mkdir(parents=True)
        self.vae_path.parent.mkdir(parents=True)
        torch.save({"model_config": {"latent_channels": 4}}, self.vae_path)
        self.source = {
            "config": self.config, "epoch": 5, "run_name": self.sd_path.parent.name,
            "source_vae": {
                "path": "/kaggle/working/experiments/" + self.name + "/vae_checkpoints/" + self.vae_path.parent.name + "/best.pt",
                "sha256": self.ns["file_sha256"](self.vae_path), "epoch": 3,
            },
            "weights": "training", "optimizer": {}, "scaler": {}, "ema": {},
            "random_state": {}, "conditioning": {}, "noise_scheduler_config": {},
        }
        torch.save(self.source, self.sd_path)
        torch.save({"weights": "ema"}, self.sd_path.with_name("best.pt"))

    def test_notebook_is_standalone_and_has_unique_cells(self):
        notebook = json.loads(NOTEBOOK.read_text())
        identifiers = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            if source.lstrip().startswith("%"):
                continue
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("experiments."))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ("train_vae", "initialize_vae"))

    def test_discovery_selects_latest_run_and_rejects_ambiguous_inputs(self):
        find = self.ns["find_sd_checkpoint"]
        self.assertEqual(find(self.input), self.sd_path)
        older = self.sd_path.parent.parent / f"{self.name}_sd-run_1"
        older.mkdir()
        shutil.copyfile(self.sd_path, older / "latest.pt")
        self.assertEqual(find(self.input), self.sd_path)
        other = self.input / "other" / "sd_checkpoints" / "another_sd-run_1"
        other.mkdir(parents=True)
        shutil.copyfile(self.sd_path, other / "latest.pt")
        with self.assertRaisesRegex(ValueError, "Più output"):
            find(self.input)
        self.assertEqual(find(self.input, self.sd_path), self.sd_path)
        with self.assertRaisesRegex(ValueError, "latest.pt"):
            find(self.input, self.sd_path.with_name("best.pt"))

    def test_vae_requires_the_original_hash(self):
        find = self.ns["find_source_vae"]
        self.assertEqual(find(self.input, self.source), self.vae_path)
        wrong = self.vae_path.parent.parent / "different_vae-run_2" / "best.pt"
        wrong.parent.mkdir()
        torch.save({"different": True}, wrong)
        with self.assertRaisesRegex(FileNotFoundError, "hash"):
            find(self.input, self.source, wrong)

    def test_target_uses_completed_epoch_and_preserves_training_settings(self):
        config = self.ns["continuation_config"](self.source, self.sd_path, self.vae_path, self.working, 10)
        self.assertEqual(config["sd_training"]["epochs"], 15)
        self.assertEqual(self.source["config"]["sd_training"]["epochs"], 100)
        self.assertEqual(config["sd_model"], self.config["sd_model"])
        self.assertEqual(config["vae_model"], self.config["vae_model"])
        self.assertEqual(config["cfg"], self.config["cfg"])
        self.assertEqual(config["sd_training"]["resume_checkpoint"], str(self.sd_path))
        self.assertEqual(self.ns["resume_settings"](config, "sd"), self.ns["resume_settings"](self.config, "sd"))
        with self.assertRaisesRegex(ValueError, "latest.pt"):
            self.ns["continuation_config"]({**self.source, "weights": "ema"}, self.sd_path, self.vae_path, self.working, 10)

    def test_final_sampling_and_evaluation_write_samples_metrics_and_report(self):
        ns = self.ns
        output_root = NOTEBOOK.resolve().parents[1] / "outputs" / "results" / self.name
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test_continuation_eval_", dir=output_root) as temporary:
            directory = Path(temporary)
            config = self.config
            config["paths"]["results"] = str(directory / "results")
            config["sampling"].update(num_samples=2, batch_size=2, inference_steps=2)
            config["evaluation"]["batch_size"] = 2
            config["data"]["image_size"] = 8
            classes = [f"Class{label:02d}" for label in range(10)]
            images, targets, files = [], [], []
            data_root = directory / "reference"
            for label, name in enumerate(classes):
                class_root = data_root / name
                class_root.mkdir(parents=True)
                for index in range(2):
                    image = torch.full((13, 8, 8), 0.1 + label * 0.03 + index * 0.01)
                    path = class_root / f"real_{index}.tif"
                    tifffile.imwrite(path, (image.permute(1, 2, 0).numpy() * 10000).astype(np.float32))
                    images.append(image)
                    targets.append(label)
                    files.append((path, label))

            class ReferenceDataset(Dataset):
                def __getitem__(self, index):
                    return images[index], targets[index]

            dataset = ReferenceDataset()
            dataset.root, dataset.classes = data_root, classes
            dataset.samples, dataset.targets = files, targets
            manifest = {
                "classes": classes,
                "files": [path.relative_to(data_root).as_posix() for path, _ in files],
                "splits": {"train": [], "validation": [], "test": list(range(20))},
            }

            class IdentityVAE(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fixed = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

                def encode(self, inputs):
                    return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: inputs))

                def decode(self, latents):
                    return SimpleNamespace(sample=latents)

            def generate(model, labels, seed, inference_steps):
                values = torch.stack([
                    torch.full((13, 8, 8), 0.1 + label * 0.03 + index * 0.01)
                    for index, label in enumerate(labels)
                ])
                values[:, 0, 0, 0], values[:, 1, 0, 0] = -0.2, 1.2
                return values

            model = TinyDenoiser()
            scheduler = DDPMScheduler(num_train_timesteps=8, prediction_type="epsilon")
            run = {
                "directory": self.sd_path.parent, "name": self.source["run_name"],
                "source_vae": self.source["source_vae"], "manifest": manifest,
            }
            torch.save({
                "model": {"adapter": torch.tensor(0.31)}, "weights": "ema",
                "source_vae": run["source_vae"], "run_name": run["name"], "epoch": 15,
            }, run["directory"] / "best.pt")
            ns.update(config=config, dataset=dataset, sd_vae=IdentityVAE(), sd_scheduler=scheduler, sample_sd_batch=generate)
            with patch.object(torch.cuda, "is_available", return_value=False), redirect_stdout(io.StringIO()):
                sampling = ns["generate_final_samples"](model, run)
                evaluated = ns["evaluate_results"](sampling, run)
            sampling_manifest = json.loads((sampling / "sampling.json").read_text())
            self.assertTrue(sampling_manifest["complete"])
            self.assertEqual(sampling_manifest["total_samples"], 20)
            image = tifffile.imread(next(sampling.glob("samples/*/MS/*.tif")))
            self.assertEqual(image.dtype, np.float32)
            self.assertEqual(image.shape, (8, 8, 13))
            self.assertLess(image.min(), 0)
            self.assertGreater(image.max(), 1)
            self.assertEqual(len(list(evaluated.glob("samples/*/MS/*_eval.tif"))), 20)
            self.assertEqual(len(list(evaluated.glob("samples/*/RGB/*_eval.png"))), 20)
            self.assertEqual(len(list(evaluated.glob("reconstructions/*/MS/*_reconstructed_eval.tif"))), 20)
            report = json.loads((evaluated / "evaluation.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["reference_split"], "test")
            self.assertEqual(report["checkpoint"]["weights"], "ema")
            self.assertLess(report["macro_metrics"]["vae_reconstruction"]["reconstruction_mae"], 1e-6)
            for name in ("metrics_per_class.csv", "spectral_curves.csv", "image_statistics.csv", "report.md", "split_manifest.json"):
                self.assertTrue((evaluated / name).is_file(), name)
            for name in ("rmc_all_classes.png", "rmc_macro.png", "metrics_comparison.png", "rgb_comparison.png"):
                self.assertTrue((evaluated / "plots" / name).is_file(), name)

    def test_ten_epochs_restore_optimizer_ema_and_resume_without_extending_target(self):
        ns = self.ns
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        self.config["data"]["num_classes"] = 2
        self.config["training"].update(batch_size=2, validation_batch_size=2)
        self.config["sd_training"].update(mixed_precision="no", ema_decay=0.9)
        images = torch.randn(4, 1, 2, 2)
        labels = torch.tensor([0, 1, 0, 1])
        dataset = TensorDataset(images, labels)
        sampler = ns["BalancedBatchSampler"]([0, 1], labels.tolist(), 2, 2, 42)
        ns["train_loader"] = DataLoader(dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(42))
        ns["validation_loader"] = DataLoader(dataset, batch_size=4, generator=torch.Generator().manual_seed(43))
        scheduler = DDPMScheduler(num_train_timesteps=8, clip_sample=False, prediction_type="epsilon")
        source_model = TinyDenoiser()
        optimizer = torch.optim.AdamW([source_model.adapter], lr=self.config["sd_training"]["learning_rate"], weight_decay=0)
        source_model.adapter.square().backward()
        optimizer.step()
        manifest = {"classes": ["a", "b"], "files": [], "splits": {"train": [0, 1], "validation": [2, 3], "test": []}}
        self.source.update({
            "model": ns["trainable_state"](source_model),
            "ema": {"adapter": source_model.adapter.detach().clone() - 0.05},
            "optimizer": optimizer.state_dict(), "global_step": 17,
            "best_validation_loss": 100.0, "split_manifest": manifest,
            "random_state": ns["get_random_state"](),
            "conditioning": {"classes": torch.ones(2, 1, 1), "null": torch.zeros(1, 1, 1)},
            "noise_scheduler_config": dict(scheduler.config),
        })
        torch.save(self.source, self.sd_path)
        config = ns["continuation_config"](self.source, self.sd_path, self.vae_path, self.working, 10)
        manifest_path = Path(config["paths"]["results"]) / "split_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest))
        ns.update(config=config, sd_conditioning=self.source["conditioning"], sd_scheduler=scheduler)
        ns["encode_sd_images"] = lambda batch, sample: batch.detach() * 0.4
        ns["validation_reference"] = lambda: {}
        previews = []

        def preview(model, run, name):
            torch.testing.assert_close(model.adapter, run["ema"]["adapter"], rtol=0, atol=0)
            previews.append(run["epoch"])
            return {"rmc_mae": 0.0}

        ns["save_sd_preview"] = preview
        run = ns["prepare_run"](config, "sd")
        run["source_vae"] = self.source["source_vae"]
        append_log = ns["append_log"]

        def interrupt_after_epoch(run, record):
            append_log(run, record)
            if record["event"] == "epoch" and record["epoch"] == 6:
                raise InterruptedError("Interruzione simulata dopo una nuova epoca completa")

        ns["append_log"] = interrupt_after_epoch
        with patch.object(torch.cuda, "is_available", return_value=False), redirect_stdout(io.StringIO()):
            with self.assertRaises(InterruptedError):
                ns["train_sd"](TinyDenoiser(), run)
            ns["append_log"] = append_log
            config = ns["continuation_config"](self.source, self.sd_path, self.vae_path, self.working, 10)
            self.assertEqual(config["sd_training"]["epochs"], 15)
            self.assertNotEqual(config["sd_training"]["resume_checkpoint"], str(self.sd_path))
            ns["config"] = config
            run = ns["prepare_run"](config, "sd")
            run["source_vae"] = self.source["source_vae"]
            model = TinyDenoiser()
            ns["train_sd"](model, run)

            latest = torch.load(run["directory"] / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(latest["epoch"], 15)
            self.assertEqual(latest["global_step"], 27)
            optimizer_steps = next(iter(latest["optimizer"]["state"].values()))["step"].item()
            self.assertEqual(optimizer_steps, 11)
            torch.testing.assert_close(model.adapter, latest["model"]["adapter"], rtol=0, atol=0)
            self.assertFalse(torch.equal(latest["model"]["adapter"], latest["ema"]["adapter"]))
            self.assertFalse(model.base.requires_grad)
            self.assertEqual(previews, list(range(6, 16)))
            best = torch.load(run["directory"] / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(best["weights"], "ema")
            before = ns["file_sha256"](run["directory"] / "best.pt")

            config = ns["continuation_config"](self.source, self.sd_path, self.vae_path, self.working, 10)
            ns["config"] = config
            rerun = ns["prepare_run"](config, "sd")
            rerun["source_vae"] = self.source["source_vae"]
            ns["train_sd"](TinyDenoiser(), rerun)
            self.assertEqual(rerun["epoch"], 15)
            self.assertEqual(rerun["global_step"], 27)
            self.assertEqual(before, ns["file_sha256"](run["directory"] / "best.pt"))

        records = [json.loads(line) for line in run["log_path"].read_text().splitlines()]
        completed = [row["epoch"] for row in records if row["event"] == "epoch"]
        self.assertEqual(completed, list(range(6, 16)))
        self.assertTrue(all(row["event"] != "start" for row in records))


if __name__ == "__main__":
    unittest.main()
