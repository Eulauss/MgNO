# MgNO README 实验复现记录 1

## 中文摘要

本次计划中的 5 个可运行 README 实验都已经跑完：Darcy smooth、Darcy rough、Navier-Stokes 1e-5、Pipe、Helmholtz。multiscale 没跑，因为缺 `mul_tri_train.mat`。正式运行从 2026-04-18 20:26:07 UTC 开始，到 2026-04-19 00:18:14 UTC 结束；tmux 会话消失是因为顺序脚本正常执行结束。

结果目录 `/nas-dev-slow/pep_design/PDE/MgNO-runs` 中，每个正式实验都有 `stdout/`、`logs/` 和 `checkpoints/`。`check_*` 目录是之前的 dry-run 检查，不属于正式复现结果。

L2 误差是相对 L2：代码里的 `LpLoss.__call__` 调用 `rel()`，即 `||prediction-target||_2 / ||target||_2`，脚本再按样本数取平均。原始 Helmholtz 运行的 H1 是例外：当时 `helm.py` 使用 `HSloss_d`，这个 H1 实现没有按 target H1 norm 归一化，因此日志里的 H1=0.04896 不是相对 H1。2026-04-19 已补充相对 H1 评估，直接重评估已有 checkpoint 得到 test Rel. H1 = 0.007959。

核心对比结果如下：

| Benchmark | 本次选用指标 | 最终 epoch 指标 | README MgNO | 结论 |
|---|---:|---:|---:|---|
| Darcy smooth | Rel. L2 = 0.001697，epoch 498 | Rel. L2 = 0.001699 | 0.0047 | 明显优于 README |
| Darcy rough | Rel. L2 = 0.003607，epoch 487，按最小 validation L2 选择 | Rel. L2 = 0.003576 | 0.0089 | 明显优于 README |
| Navier-Stokes 1e-5 | `test_l2_full` = 0.01597，epoch 493 | `test_l2_full` = 0.01598 | 0.0820 | 明显优于 README |
| Pipe flow | Rel. L2 = 0.002955，epoch 473 | Rel. L2 = 0.002981 | 0.0183 | 明显优于 README |
| Helmholtz | post-hoc Rel. H1 = 0.007959，旧日志 H1 = 0.04896 | 旧日志 H1 = 0.04896 | 0.0284 | 相对 H1 优于 README；旧日志不可直接比较 |

结论：实验产物完整，整体运行符合预期；但数值不能说和 README 完全对上。非 Helmholtz 的四个实验显著好于 README 表格；Helmholtz 原始日志的 H1 口径错误，按相对 H1 重新评估已有 checkpoint 后也优于 README。不过 Helmholtz 当时的训练 objective 仍然是旧的绝对 H1，若要严格复现“相对 H1 训练”的版本，需要只重跑 Helmholtz。

## Scope

This run reproduced the README-listed experiments that had available data:

- Darcy flow, smooth
- Darcy flow, rough
- Navier-Stokes, viscosity/Re label `1e-5` as used by the repo
- Pipe flow
- Helmholtz

Darcy multiscale was not run because `mul_tri_train.mat` was not available. The Google Drive files found for it returned owner/editor-only permission errors during dataset preparation.

## Environment And Paths

- Date range: 2026-04-18 20:26:07 UTC to 2026-04-19 00:18:14 UTC
- Conda environment: `fno`
- Device: `cuda:1`
- Data root: `/vepfs-dev/tianzt/pde_data`
- Run root: `/nas-dev-slow/pep_design/PDE/MgNO-runs`
- Scheduler log: `/nas-dev-slow/pep_design/PDE/MgNO-runs/scheduler/20260418_162607.runner.log`
- Runner script: `scripts/run_readme_experiments.sh`
- Scheduler script: `scripts/schedule_readme_experiments_tmux.sh`

The training scripts were run with the README hyperparameters plus these path/device arguments:

```bash
--data_root /vepfs-dev/tianzt/pde_data
--run_root /nas-dev-slow/pep_design/PDE/MgNO-runs
--device cuda:1
```

The output layout is one directory per experiment:

```text
/nas-dev-slow/pep_design/PDE/MgNO-runs/<experiment>/
  checkpoints/
  logs/
  stdout/
```

There are also `check_*` directories under the run root. Those are dry-run validation logs from before the scheduled run and are not formal experiment results.

## Completion Status

The scheduled tmux job is no longer present because the runner completed and exited. All five scheduled experiments reached their configured final epoch and wrote a checkpoint.

| Experiment | Started UTC | Ended UTC | Epochs | Checkpoint |
|---|---:|---:|---:|---|
| Darcy smooth | 2026-04-18 20:26:07 | 2026-04-18 21:02:59 | 500/500 | `darcy_smooth/checkpoints/MgNO_DC_smooth_darcy_20260418_202612.pt` |
| Darcy rough | 2026-04-18 21:02:59 | 2026-04-18 21:48:04 | 500/500 | `darcy_rough/checkpoints/MgNO_DC_darcy20c6_20260418_210304.pt` |
| Navier-Stokes 1e-5 | 2026-04-18 21:48:04 | 2026-04-18 22:44:25 | 500/500 | `navier_stokes_1e-5/checkpoints/MgNO_1e-5_20260418_214809.pt` |
| Pipe | 2026-04-18 22:44:25 | 2026-04-18 23:54:33 | 500/500 | `pipe/checkpoints/MgNO_DC_pipe_20260418_224430.pt` |
| Helmholtz | 2026-04-18 23:54:33 | 2026-04-19 00:18:14 | 100/100 | `helmholtz/checkpoints/MgNO_helm_helm_20260418_235438.pt` |

## Metric Semantics

For the L2 metrics, the code uses `LpLoss.__call__`, which calls `LpLoss.rel`. This computes per-sample relative L2:

```text
||prediction - target||_2 / ||target||_2
```

The training scripts instantiate `LpLoss(size_average=False)` and then divide the accumulated sum by the number of samples, so the logged L2 values are average relative L2 errors.

For Darcy and pipe H1, the code uses `HsLoss`, whose `rel` method also normalizes by the target norm in Fourier-weighted Sobolev space. For the original Helmholtz run, however, the script switched to `HSloss_d`; that implementation computes a finite-difference absolute-style quantity and does not divide by the target H1 norm. Therefore the original Helmholtz H1 value is not a relative H1.

After the 2026-04-19 follow-up check, `HSloss_d_relative` was added and the saved Helmholtz checkpoint was re-evaluated without retraining:

```bash
conda run -n fno python scripts/eval_helmholtz_checkpoint.py \
  --checkpoint /nas-dev-slow/pep_design/PDE/MgNO-runs/helmholtz/checkpoints/MgNO_helm_helm_20260418_235438.pt \
  --data_root /vepfs-dev/tianzt/pde_data \
  --device cuda:1 \
  --batch_size 10
```

Corrected Helmholtz checkpoint metrics:

| Split | Rel. L2 | Rel. H1, weight 1 | Rel. H1, weight 10 | Legacy logged H1 |
|---|---:|---:|---:|---:|
| test | 0.0080878 | 0.0079591 | 0.0079596 | 0.0489629 |
| val | 0.0086255 | 0.0084941 | 0.0084946 | 0.0529321 |

For Navier-Stokes, the log contains several L2 values:

- `test_l2_step`: average relative L2 accumulated per autoregressive step.
- `test_l2_full`: relative L2 over the full autoregressive rollout tensor.
- `test_l2_full_2`: first-step relative L2 recorded during rollout.

The table below compares README's single "Rel. L2" number against `test_l2_full`, because that is the full rollout metric and is the closest single Navier-Stokes test metric in the script.

## Results Compared With README

README reports these MgNO metrics:

| Benchmark | README metric | README MgNO |
|---|---:|---:|
| Darcy smooth | Rel. L2 | 0.0047 |
| Darcy rough | Rel. L2 | 0.0089 |
| Darcy multiscale | Rel. L2 | 0.0986 |
| Navier-Stokes 1e-5 | Rel. L2 | 0.0820 |
| Pipe flow | Rel. L2 | 0.0183 |
| Helmholtz | H1 | 0.0284 |

Our parsed results:

| Benchmark | Selected metric from this run | Final epoch metric | README MgNO | Comparison |
|---|---:|---:|---:|---|
| Darcy smooth | Rel. L2 = 0.001697 at epoch 498 | Rel. L2 = 0.001699 | 0.0047 | Better than README |
| Darcy rough | Rel. L2 = 0.003607 at epoch 487, selected by min validation L2 | Rel. L2 = 0.003576 | 0.0089 | Better than README |
| Navier-Stokes 1e-5 | `test_l2_full` = 0.01597 at epoch 493 | `test_l2_full` = 0.01598 | 0.0820 | Better than README |
| Pipe flow | Rel. L2 = 0.002955 at epoch 473 | Rel. L2 = 0.002981 | 0.0183 | Better than README |
| Helmholtz | post-hoc Rel. H1 = 0.007959; legacy H1 = 0.04896 at epoch 100 | legacy H1 = 0.04896 | 0.0284 | Corrected relative H1 is better than README; legacy H1 is not comparable |

Additional Navier-Stokes final epoch values:

| Navier metric | Final value | Best value in run |
|---|---:|---:|
| `test_l2_step` | 0.01220 | 0.01220 at epoch 497/498/500-level rounding |
| `test_l2_full` | 0.01598 | 0.01597 at epoch 493 |
| `test_l2_full_2` | 0.008965 | 0.008965 at epoch 500 |

## Interpretation

The run completed successfully and produced the expected artifacts. The outputs are not numerically identical to the README table.

The non-Helmholtz experiments are substantially better than the README MgNO values. That is a large difference, not just run-to-run noise. Possible reasons include the current repository scripts, data split/evaluation protocol, or table values being from the paper rather than exactly these scripts. The current scripts also do not set a random seed, so exact reproducibility is not guaranteed.

Helmholtz originally looked worse only because the logged H1 used the legacy `HSloss_d` quantity. Re-evaluating the saved checkpoint with relative finite-difference H1 gives test Rel. H1 = 0.007959, which is better than the README value 0.0284. This is a post-hoc metric correction; the checkpoint itself was still trained with the legacy absolute H1 objective.

## Source Logs

Formal run logs:

- `/nas-dev-slow/pep_design/PDE/MgNO-runs/darcy_smooth/logs/MgNO_DC_smooth_darcy_20260418_202612.log`
- `/nas-dev-slow/pep_design/PDE/MgNO-runs/darcy_rough/logs/MgNO_DC_darcy20c6_20260418_210304.log`
- `/nas-dev-slow/pep_design/PDE/MgNO-runs/navier_stokes_1e-5/logs/MgNO_1e-5_20260418_214809.log`
- `/nas-dev-slow/pep_design/PDE/MgNO-runs/pipe/logs/MgNO_DC_pipe_20260418_224430.log`
- `/nas-dev-slow/pep_design/PDE/MgNO-runs/helmholtz/logs/MgNO_helm_helm_20260418_235438.log`
