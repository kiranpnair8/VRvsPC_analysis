# Frozen REVE Genuine/Impostor Geometry

`analyze_reve_genuine_impostor_geometry_cv5.py` measures distances in the original
512-dimensional frozen REVE space. It does not use MLP/SVM/EEGNet scores and does
not train a model, rebuild data, generate folds, or run REVE inference.

## Inputs and Enrollment

The authoritative indices are `folds_json` inside
`out/cv5/cv5_verification_dataset_lphp10_50.npz`. Subjects are recovered from
`y_epoch_pc`, `y_epoch_vr`, and `subject_map`. The saved genuine trial metadata is
checked against every test fold. Each subject/environment must have 120 epochs,
partitioned into 78 training, 18 validation and 24 test epochs in each fold;
every physical epoch must appear in the test set exactly once across five folds.

By default the script loads
`reve_embedding_analysis/reve_unique_embeddings.npz`. Override with
`--embeddings_npz PATH`. If that consolidated file is absent, it calls the existing
`reconstruct_unique_embeddings()` helper to read genuine rows from the ten
PC->PC / VR->VR outer-test caches in `out/cv5/cache/reve`. No embeddings are
recomputed. Actual source files and hashes are recorded in the manifest (and
cache-manifest CSV when reconstructing). Existing legacy caches contain only Z;
their row alignment relies on their matching saved CV5 trial arrays, as in the
previous analysis. Shapes/IDs/fold membership are checked, but this cannot prove
encoder-weight provenance from a cache containing no weight hash.

Because REVE is frozen and extraction is per epoch, a unique embedding originally
cached when an epoch was tested in one fold can also be indexed as training data
in a different fold. It contributes to enrollment only when that epoch's saved
partition is training in the current fold. Validation and current-fold test
embeddings never enter a centroid. No global PCA, scaling or 120-epoch centroid
is used for enrollment. Centroids average 78 source PC or VR epochs, or 156
concatenated PC+VR epochs for full mixed enrollment.

Full mixed enrollment has twice the training budget. Any observed recovery here
does not isolate environmental diversity from the additional enrollment data;
the separate Mixed-Matched experiment addresses that distinction.

Each target test epoch is compared with all 21 enrollment centroids: one genuine
and 20 impostor distances. This gives 504 genuine and 10,080 impostor comparisons
per fold/scenario, and 317,520 comparison rows across six scenarios/five folds.
Epoch indices are environment-global indices in the reconstructed CV5 dataset,
not raw recording timestamps. Subject labels preserve the encoded IDs; names
are available through the manifest's original `subject_map`.

## Metrics and Statistics

Lower distance is more genuine-like. Euclidean distance is the L2 norm; cosine
distance is `1 - dot(a,b)/(norm(a)*norm(b))`. Zero/zero has cosine distance 0;
zero/nonzero has distance 1. Zero norms are counted in the audit CSV.

`d_prime = (mean_impostor - mean_genuine) /
sqrt(0.5 * (sample_var_genuine + sample_var_impostor))`. Larger positive values
mean stronger separation. The saved Cohen-style separation is identical to this
equal-weight variance definition, rather than the sample-size-pooled Cohen d.
Zero denominator yields NaN and is not silently interpreted as separation.

AUC and analytical EER use negative distance as score, labels 1 genuine/0 impostor.
EER follows the project's closest-FAR/FRR unique-score convention, without a
quadratic threshold scan. These are embedding geometry measurements, not the
authentication classifiers' actual EER or operational thresholds.

Fold summaries include mean, sample SD, median, d-prime, margin, AUC, EER and
histogram overlap. Histogram overlap is the sum of minimum bin probabilities,
using 80 shared equal-width bins over all scenarios for each distance metric.
Plots pool outer-held-out observations only for visualization. Impostor pairs
and repeated epochs are never treated as independent inferential units.

Participant summaries group by the true subject of the probe, comparing its
24 genuine distances and 480 nonmatching-centroid distances per fold. Every
participant contributes one arithmetic average across five folds before paired
testing (n=21). At each fixed target environment, compare Matched-Cross,
Cross-Mixed and Matched-Mixed, for genuine distance and margin. Shapiro-Wilk on
paired differences selects a paired t-test if p >= .05, otherwise two-sided
Wilcoxon signed-rank (zeros dropped). Effect sizes are paired Cohen dz or signed
rank-biserial correlation. Constant differences use Wilcoxon; identical pairs
have p=1 and effect=0. Holm correction covers six comparisons (three x two
targets) within each distance/outcome family; all signs are first minus second.

The d-prime figure uses means of participant five-fold averages with percentile
95% CIs from 10,000 participant resamples, seed 42. Those participant means may
differ from pooled fold d-prime because d-prime is a nonlinear statistic.

## CPU Command

```bash
python analyze_reve_genuine_impostor_geometry_cv5.py \
  --cv5_npz ./out/cv5/cv5_verification_dataset_lphp10_50.npz \
  --seed 42 \
  --out_dir ./results/reve_geometry_cv5 \
  --fig_dir ./figures/reve_geometry_cv5 \
  --distance both
```

Use the existing scientific Conda environment. No CUDA, torch, transformers,
model loading, or GPU allocation is needed. `--skip_plots` saves only tables;
`--distance euclidean` / `cosine` restricts the analysis. When both are selected,
the long CSV contains both distance columns. Single-metric runs save only that
distance column. Outputs are never overwritten: choose fresh directories for
a rerun. Mixed-Matched is intentionally excluded from this primary analysis.

## Outputs

Under `results/reve_geometry_cv5/`:

- `reve_embedding_trial_distances_cv5.csv`: every epoch/claimed-identity pair.
- `reve_embedding_separation_per_fold_cv5.csv`: 60 fold/scenario/metric rows.
- `reve_embedding_separation_summary_cv5.csv`: five-fold means and sample SDs.
- `reve_embedding_participant_fold_summary_cv5.csv`: 1,260 subject/fold rows.
- `reve_embedding_participant_summary_cv5.csv`: 252 subject five-fold averages.
- `reve_embedding_statistical_tests_cv5.csv`: 24 planned participant tests.
- `reve_embedding_enrollment_indices_cv5.csv`: exact training centroid membership.
- `reve_embedding_sanity_checks_cv5.csv`: 30 fold/scenario audits.
- `reve_embedding_dprime_ci_cv5.csv`: 12 participant bootstrap CIs when plotting.
- `reve_geometry_enrollment_centroids.npz`: centroids with fold/scenario/subject.
- `reve_geometry_cv5_manifest.json`: provenance, formulas, histogram edges.
- `reve_embedding_cache_manifest_cv5.csv`: only when reconstructing from caches.

Under `figures/reve_geometry_cv5/`, vector PDF and 400-dpi PNG:

- `reve_genuine_impostor_euclidean_cv5`: six distribution panels, PC test top row.
- `reve_genuine_impostor_cosine_cv5`: the same layout for cosine distance.
- `reve_embedding_dprime_cv5`: Euclidean/cosine panels; participant mean +/- CI.
- `reve_participant_genuine_distance_shift_cv5`: one line per participant;
  fixed-target Matched/Cross/Mixed comparisons in a 2x2 layout.
- `reve_participant_margin_cv5`: same layout for impostor-minus-genuine margin.

The genuine-distance paired figure is a strong main-paper candidate because it
directly shows enrollment mismatch and possible mixed recovery within subjects.
Use the distribution and margin figures in the appendix; the final choice should
follow the observed results. Evidence concerns frozen-REVE representation
geometry and does not establish that REVE causes authentication degradation.

Synthetic validation: `python -m unittest test_reve_geometry_cv5 -v`.

