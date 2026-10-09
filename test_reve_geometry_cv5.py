"""Focused synthetic checks; fixtures never represent actual paper results."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

import analyze_reve_genuine_impostor_geometry_cv5 as geometry
from cv5_metrics import compute_eer
from cv5_split_utils import FoldConfig, generate_epoch_fold_indices


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(os.environ.get("GEOMETRY_SMOKE_DIR") or tempfile.mkdtemp(prefix="reve_geometry_test_"))
        cls.directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        cls.subjects = np.arange(21)
        cls.y = np.repeat(cls.subjects, 120)
        cls.folds = generate_epoch_fold_indices({"pc": cls.y, "vr": cls.y}, FoldConfig())
        identity = rng.normal(size=(21, 512)) * .7
        shift = rng.normal(size=512) * .25
        Zpc = identity[cls.y] + rng.normal(size=(2520, 512)) * .4
        Zvr = identity[cls.y] + shift + rng.normal(size=(2520, 512)) * .4
        arrays = {"Z": np.concatenate([Zpc, Zvr]).astype(np.float32),
                  "subject_label": np.tile(cls.y, 2),
                  "environment": np.repeat(["PC", "VR"], 2520),
                  "environment_epoch": np.tile(np.arange(2520), 2),
                  "subject_epoch": np.tile(np.arange(120), 42),
                  "outer_test_fold": np.zeros(5040, dtype=int)}
        dataset = {"meta_json": json.dumps({"n_folds": 5, "subject_map": {f"subject_{s}": int(s) for s in cls.subjects}}),
                   "folds_json": json.dumps(cls.folds), "y_epoch_pc": cls.y, "y_epoch_vr": cls.y}
        for ei, env in enumerate(("pc", "vr")):
            for fold in range(5):
                indices = np.concatenate([cls.folds[env][s][fold]["test"] for s in cls.subjects])
                arrays["outer_test_fold"][ei * 2520 + indices] = fold
                prefix = f"{env}_fold{fold}_test"
                dataset[f"y_{prefix}"] = np.ones(504, dtype=int)
                dataset[f"cid_{prefix}"] = cls.y[indices]
                dataset[f"trial_source_epoch_{prefix}"] = indices
        np.savez_compressed(cls.directory / "cv5.npz", **dataset)
        np.savez_compressed(cls.directory / "unique.npz", **arrays)
        cls.arrays = arrays
        cls.args = type("Args", (), dict(cv5_npz=cls.directory / "cv5.npz",
                                        embeddings_npz=cls.directory / "unique.npz"))()

    def test_distance_definitions_and_zero_norms(self):
        values = geometry.distance_matrices(np.array([[1., 0.], [0., 0.]]),
                                           np.array([[1., 0.], [-1., 0.], [0., 0.]]),
                                           ["euclidean", "cosine"])
        np.testing.assert_allclose(values["euclidean"], [[0, 2, 1], [1, 1, 0]])
        np.testing.assert_allclose(values["cosine"], [[0, 2, 1], [1, 1, 0]])

    def test_eer_matches_existing_helper_with_ties(self):
        for genuine, impostor in (([0., 0., 2., 3.], [1., 1., 2., 4.]), ([1., 1.], [1., 1.]),
                                  ([0., .1], [2., 3.])):
            actual = geometry.separation(np.array(genuine), np.array(impostor), np.linspace(0, 4, 10))
            expected = compute_eer(-np.r_[genuine, impostor], np.r_[np.ones(len(genuine)), np.zeros(len(impostor))])
            self.assertAlmostEqual(actual["embedding_space_eer"], expected)
        separated = geometry.separation(np.array([0., .1]), np.array([2., 3.]), np.linspace(0, 4, 10))
        self.assertEqual(separated["roc_auc"], 1.)
        self.assertGreater(separated["d_prime"], 0)

    def test_wrong_test_fold_rejected(self):
        subjects, bank, folds, meta, _, _ = geometry.load_inputs(self.args)
        broken = {key: value.copy() for key, value in self.arrays.items()}
        broken["outer_test_fold"][0] = (broken["outer_test_fold"][0] + 1) % 5
        tests = {(env.upper(), k): np.concatenate([folds[env][str(s)][str(k)]["test"] for s in subjects])
                 for env in ("pc", "vr") for k in range(5)}
        with self.assertRaisesRegex(AssertionError, "outer_test_fold"):
            geometry.validate_inputs(broken, {"PC": self.y, "VR": self.y}, folds, meta, tests)

    def test_full_cardinality_training_only_centroids_and_figures(self):
        subjects, bank, folds, _, _, _ = geometry.load_inputs(self.args)
        with contextlib.redirect_stdout(io.StringIO()):
            observations, enrollment, saved, audit = geometry.construct_distances(subjects, bank, folds, ["euclidean", "cosine"])
        self.assertEqual(len(observations), 317520)
        self.assertEqual(len(saved), 630)
        self.assertTrue((audit.genuine_count == 504).all())
        self.assertTrue((audit.impostor_count == 10080).all())
        for fold, scenario, subject, centroid in saved:
            sources, _ = geometry.SCENARIOS[scenario]
            rows = [bank[env]["Z"][geometry.split_indices(folds, env, subject, fold, "train")] for env in sources]
            np.testing.assert_allclose(centroid, np.concatenate(rows).mean(axis=0))
        for target, scenarios in geometry.TARGET_SCENARIOS.items():
            for fold in range(5):
                probe_sets = [observations.loc[(observations.scenario == s) & (observations.fold == fold), "test_epoch_index"].to_numpy()
                              for s in scenarios]
                for probe_set in probe_sets[1:]:
                    np.testing.assert_array_equal(probe_sets[0], probe_set)
        metrics = ["euclidean", "cosine"]
        fold_frame, subject_folds, participants, summary, edges = geometry.summarize(observations, metrics, 80)
        self.assertEqual(len(fold_frame), 60)
        self.assertEqual(len(subject_folds), 1260)
        self.assertEqual(len(participants), 252)
        tests = geometry.paired_statistics(participants, metrics, 42)
        self.assertEqual(len(tests), 24)
        self.assertTrue((tests.holm_p >= tests.raw_p - 1e-14).all())
        args = type("Args", (), dict(fig_dir=self.directory / "figures", out_dir=self.directory / "results",
                                    dpi=400, seed=42, bootstrap_repeats=1000, skip_plots=False))()
        args.fig_dir.mkdir(exist_ok=True)
        ci = geometry.plot_all(observations, participants, metrics, edges, args)
        self.assertEqual(len(ci), 12)
        for stem in geometry.figure_stems(metrics):
            for suffix in ("pdf", "png"):
                self.assertTrue((args.fig_dir / f"{stem}.{suffix}").is_file())
        with self.assertRaises(FileExistsError):
            geometry.guard_outputs(args, metrics)
        print(f"Synthetic preview directory: {args.fig_dir}")


if __name__ == "__main__":
    unittest.main()

