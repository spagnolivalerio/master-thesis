"""Controlli numerici e smoke test della pipeline, senza dataset EuroSAT."""

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.latent_unmixing_per_class_softmax import (
    evaluation,
    sample,
    train_ddpm,
    train_encoder,
    utils,
)


class TrainingInterrupted(Exception):
    """Simula l'arresto dopo che latest è stato scritto integralmente."""


class SoftmaxExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_logit_round_trip_and_simplex(self):
        generator = torch.Generator().manual_seed(3)
        labels = torch.tensor([1, 0, 1])
        logits = torch.randn(3, 4, 2, 2, generator=generator)
        statistics = {
            "mean": [[0.1, 1.0, -2.0, 0.4], [2.0, -0.3, 0.5, 1.2]],
            "standard_deviation": [[0.5, 2.0, 1.0, 3.0], [2.0, 0.2, 3.0, 0.5]],
            "temperature": 0.7,
        }
        expected = torch.softmax(logits / statistics["temperature"], dim=1)
        latent = utils.to_latents(logits, labels, statistics)
        actual = utils.from_latents(latent, labels, statistics)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(1), torch.ones(3, 2, 2))
        shifted = utils.softmax_abundances(logits + 10, statistics["temperature"])
        torch.testing.assert_close(shifted, expected)
        extreme = torch.tensor([10000.0, -10000.0, 0.0]).view(1, 3, 1, 1)
        projected = utils.softmax_abundances(extreme, 1.0)
        self.assertTrue(torch.isfinite(projected).all())
        self.assertTrue((projected > 0).all())
        torch.testing.assert_close(projected.sum(1), torch.ones(1, 1, 1))

    def test_statistics_are_per_class_and_channel(self):
        class FixedEncoder(torch.nn.Module):
            def encode(self, images, labels):
                return images, utils.softmax_abundances(images, 1.0)

        config = utils.load_config()
        config["data"]["num_classes"] = 2
        config["model"]["num_bases"] = 3
        images = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
        labels = torch.tensor([1, 0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=3)
        statistics = utils.compute_logit_statistics(
            FixedEncoder(), loader, config, torch.device("cpu")
        )
        for label in range(2):
            values = images[labels == label].permute(1, 0, 2, 3).flatten(1)
            torch.testing.assert_close(torch.tensor(statistics["mean"][label]), values.mean(1))
            torch.testing.assert_close(
                torch.tensor(statistics["standard_deviation"][label]),
                values.std(1, correction=0),
            )
        self.assertEqual(statistics["latent_channels"], 3)
        self.assertEqual(statistics["temperature"], config["model"]["abundance_temperature"])

    def test_training_resume_sampling_and_evaluation(self):
        config = copy.deepcopy(utils.load_config())
        root = Path(__file__).resolve().parents[2]
        experiment = Path(__file__).resolve().parent
        name = config["experiment"]["name"]
        with ExitStack() as stack:
            locations = {
                "results": root / "outputs" / "results" / name,
                "previews": root / "outputs" / "previews" / name,
                "encoder_checkpoints": experiment / "encoder_checkpoints",
                "ddpm_checkpoints": experiment / "ddpm_checkpoints",
                "logs": experiment / "log",
            }
            temporary = {}
            for key, directory in locations.items():
                directory.mkdir(parents=True, exist_ok=True)
                temporary[key] = Path(stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="test_", dir=directory)
                ))
            config["paths"].update({
                "results": str(temporary["results"] / "generated"),
                "encoder_checkpoints": str(temporary["encoder_checkpoints"]),
                "ddpm_checkpoints": str(temporary["ddpm_checkpoints"]),
                "encoder_logs": str(temporary["logs"] / "encoder"),
                "ddpm_logs": str(temporary["logs"] / "ddpm"),
                "encoder_previews": str(temporary["previews"] / "encoder"),
                "ddpm_previews": str(temporary["previews"] / "ddpm"),
            })
            config["data"]["root"] = str(temporary["results"] / "dataset")
            config["data"]["image_size"] = 8
            config["model"].update(
                num_bases=3, hidden_channels=8, num_res_blocks=1,
                norm_num_groups=2, abundance_temperature=0.7,
            )
            config["ddpm_model"].update(
                base_channels=8, channel_multipliers=[1, 2],
                num_res_blocks=1, norm_num_groups=4, attention_head_dim=4,
            )
            config["diffusion"]["timesteps"] = 8
            config["ddpm_training"].update(warmup_steps=0, preview_inference_steps=2)
            config["evaluation"].update(images_per_class=1, batch_size=2, inference_steps=2)
            for section in ("encoder_training", "ddpm_training"):
                config[section].update(
                    epochs=2, num_workers=0, pin_memory=False,
                    mixed_precision="no", resume=True,
                )

            generator = np.random.default_rng(42)
            for label in range(config["data"]["num_classes"]):
                directory = Path(config["data"]["root"]) / f"Class{label:02d}"
                directory.mkdir(parents=True)
                for index in range(20):
                    image = generator.integers(100, 8000, size=(8, 8, 13), dtype=np.uint16)
                    tifffile.imwrite(directory / f"image_{index:02d}.tif", image)

            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            for module in (train_encoder, train_ddpm, sample, evaluation):
                stack.enter_context(patch.object(module, "load_config", return_value=config))

            real_save = torch.save

            def save_then_interrupt(checkpoint, path):
                real_save(checkpoint, path)
                if str(path).endswith("-latest.pt") and checkpoint["epoch"] == 1:
                    raise TrainingInterrupted

            for module, stage in ((train_encoder, "encoder"), (train_ddpm, "ddpm")):
                with patch.object(torch, "save", side_effect=save_then_interrupt):
                    with self.assertRaises(TrainingInterrupted):
                        module.main()
                best_path = module.main()
                self.assertTrue(best_path.is_file())
                latest_path = utils.latest_checkpoint(
                    config["paths"][f"{stage}_checkpoints"], f"{name}_{stage}"
                )
                checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["epoch"], 2)
                self.assertEqual(checkpoint["run_name"], f"{name}_{stage}-run_1")
                self.assertGreater(checkpoint["global_step"], 0)
                for key in ("optimizer", "scheduler", "scaler", "random_state", "config"):
                    self.assertIn(key, checkpoint)
                log_path = Path(config["paths"][f"{stage}_logs"]) / f"{checkpoint['run_name']}.jsonl"
                records = [json.loads(line) for line in log_path.read_text().splitlines()]
                self.assertTrue(any(record["event"] == "resume" for record in records))
                batches = [record for record in records if record["event"] == "batch"]
                self.assertEqual(len(batches), checkpoint["global_step"])
                self.assertEqual({record["epoch"] for record in batches}, {1, 2})
                for key in ("loss", "mse_loss", "loss_mean", "learning_rate", "gradient_norm", "throughput", "peak_vram_mb"):
                    self.assertIn(key, batches[-1])
                if stage == "ddpm":
                    self.assertIn("ema", checkpoint)
                    self.assertNotIn("ilr_stats", checkpoint)
                    self.assertEqual(checkpoint["logit_stats"]["latent_channels"], 3)
                    self.assertEqual(checkpoint["logit_stats"]["temperature"], 0.7)
                    best = torch.load(best_path, map_location="cpu", weights_only=False)
                    self.assertEqual(best["weights"], "ema")
                    encoder_best = utils.latest_checkpoint(
                        config["paths"]["encoder_checkpoints"], f"{name}_encoder", "best"
                    )
                    source = torch.load(encoder_best, map_location="cpu", weights_only=False)
                    for key, value in source["model"].items():
                        torch.testing.assert_close(checkpoint["unmixing_model"][key], value, rtol=0, atol=0)

            with patch("sys.argv", ["sample", "--n", "1"]):
                sample.main()
            sampled = Path(config["paths"]["results"]) / f"{name}-sampling_1-run_1" / "samples"
            self.assertEqual(len(list(sampled.glob("*/MS/*.tif"))), 10)
            self.assertEqual(len(list(sampled.glob("*/RGB/*.png"))), 10)
            image = tifffile.imread(sampled / "Class00" / "MS" / "Class00_0001.tif")
            self.assertEqual(image.dtype, np.float32)
            self.assertEqual(image.shape, (8, 8, 13))
            self.assertTrue(np.isfinite(image).all() and (image >= 0).all())

            with patch("sys.argv", ["sample", "--class", "Class03", "--n", "2"]):
                sample.main()
            selected = Path(config["paths"]["results"]) / f"{name}-sampling_2-run_1" / "samples"
            self.assertEqual([path.name for path in selected.iterdir()], ["Class03"])
            self.assertEqual(len(list(selected.glob("*/MS/*.tif"))), 2)

            with patch("sys.argv", ["evaluation"]):
                evaluated = evaluation.main()
            self.assertTrue((evaluated / "plots" / "rmc_all_classes.png").is_file())
            self.assertEqual(len(list(evaluated.glob("samples/*/MS/*_eval.tif"))), 10)
            report = json.loads((evaluated / "evaluation.json").read_text())
            self.assertEqual(len(report["metrics"]), 10)
            self.assertTrue(np.isfinite(list(report["macro"].values())).all())
            with patch("sys.argv", ["evaluation", "--path", str(sampled)]):
                evaluation.main()


if __name__ == "__main__":
    unittest.main()
