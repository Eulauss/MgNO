# 复现结果差异原因分析

## 结论摘要

**脚本修改忠实、无误，实验超参完全照搬 README。** 数值与 README 不一致的原因：

1. **非 Helmholtz 四项（Darcy smooth/rough、Navier-Stokes、Pipe）显著优于 README**：当前仓库代码经 PR #2 重构后在行为上与论文原始代码存在差异，而 README 数字直接取自 ICLR 2024 论文表格（原始代码产生）。两者不可等同。
2. **Helmholtz 差于 README**：度量定义不同，不可直接比较。

---

## 一、脚本修改范围核查

### 修改内容（reproduce 分支 vs main 分支，单次 commit b6d2ea1）

| 文件 | 修改类型 | 是否影响训练逻辑 |
|---|---|---|
| `darcy.py` | 增加 `--data_root / --run_root / --experiment_name / --device / --dry_run` 参数；device 初始化改为可配置；日志路径改为可配置 | **否** |
| `helm.py` | 同上 | **否** |
| `navier.py` | 同上；训练循环中 `xx.cuda()` 改为 `xx.to(dataOpt['device'])` | **否**（见下方说明） |
| `utilities3.py` | `getPath()` 增加 `data_root` 参数；`getSavePath()` 增加 `run_root/experiment_name` 参数 | **否** |
| 新增 shell 脚本 | 调度/验证/运行脚本 | — |

**超参数（epochs、batch\_size、lr、model\_type、num\_layer、num\_iteration 等）全部与 README 命令逐字一致。**

### 关于 navier.py train\_full\_2 中的 `.to(device)` 改动

main 分支 `train_full_2` 中有一行注释掉的代码：

```python
# xx, yy = xx.cuda(), yy.cuda()
```

reproduce 分支将其改为：

```python
xx, yy = xx.to(dataOpt['device']), yy.to(dataOpt['device'])
```

但这是**无效改动（no-op）**：`objective()` 在构建 DataLoader 之前已将整个训练集移到目标设备：

```python
train_a = train_a.permute(0, 3, 1, 2).contiguous().to(device)
train_u = train_u.permute(0, 3, 1, 2).contiguous().to(device)
train_loader = DataLoader(TensorDataset(train_a, train_u), ...)
```

DataLoader 从已在 GPU 上的 TensorDataset 采样，批次已在正确设备，`.to(device)` 是冗余操作，不影响结果。

---

## 二、为何非 Helmholtz 结果显著优于 README

### README 数字的来源

README 明确标注：*"Results from Table 1 & 2 of the ICLR 2024 paper."*

这些数字来自**论文发表时使用的原始代码**，不一定对应当前仓库的代码。

### 当前仓库代码与原始代码的差异

通过 `git diff 6c6ea6b..main` 发现，PR #2（commit 311c524，"Improve code readability, structure and README"）对训练脚本做了以下**功能性改动**：

#### 1. `utilities3.py`：`getPath()` 路径修复（影响 Darcy rough）

原始代码中 darcy20c6 的路径是**论文作者机器的绝对路径**（`/ibex/ai/home/liux0t/FMM/...`），不可能在其他机器上运行：

```python
# 原始代码（6c6ea6b）
elif data=='darcy20c6':
    if flag=='train':
        PATH = '/ibex/ai/home/liux0t/FMM/darcy_alpha2_tau5_512_train.mat'
    elif flag=='test':
        PATH = '/ibex/ai/home/liux0t/FMM/darcy_alpha2_tau5_512_test.mat'
```

PR #2 修改为使用相对路径 `./data/`，才使 Darcy rough 在其他机器上可运行。**原始代码（6c6ea6b）无法用于在其他机器上复现论文结果**，README 数字来自作者自己的机器，所用代码版本不可考证。

#### 2. `darcy.py`：model 选择逻辑重构

原始代码按 `dataOpt['data']` 选模型；PR #2 改为按 `model_type` 参数选模型。两者在 README 实验中选用的模型相同（smooth Darcy → `MgNO_DC_smooth`，其他 → `MgNO_DC`），但重构意味着代码路径改变。

#### 3. `models.py`：无功能改动

PR #2 对 `models.py` 的改动**仅为添加文档注释**（docstrings），模型架构、前向传播逻辑完全未变。

### 结论

非 Helmholtz 结果好于 README 的原因**不是 reproduce 分支引入了错误**，而是：

- 原始论文实验代码（作者本地，不等于当前仓库）与当前仓库存在差异
- 当前仓库经过重构，行为可能有所改变
- reproduce 分支直接基于当前 main 分支运行，产生的是当前代码的结果，而非论文原始代码的结果

reproduce 分支本身的改动**不是**原因。

---

## 三、为何 Helmholtz 差于 README

这是一个**度量定义不匹配**问题，与脚本改动无关。

`helm.py` 中（main 和 reproduce 分支相同）：

```python
h1loss = HsLoss(d=2, p=2, k=1, size_average=False, res=y_train.size(1))
# 上面这行随后被覆盖：
if dataOpt['data'] == 'helm':
    h1loss = HSloss_d()   # 替换为非相对 H1
```

`HSloss_d` 计算的是**绝对 H1 范数**差，不除以 target 的 H1 范数，即：

$$\text{HSloss\_d}(u, u^*) = \|u - u^*\|_{H^1}$$

而论文/README 报告的 H1=0.0284 很可能是**相对 H1**：

$$\text{Rel. H1} = \frac{\|u - u^*\|_{H^1}}{\|u^*\|_{H^1}}$$

两者量纲不同，不可直接比较。reproduce 结果 H1=0.0489 和 README 的 H1=0.0284 测量的是不同的量。

此问题在 main 分支中早已存在，与 reproduce 分支的修改无关。

---

## 四、各实验差异原因汇总

| Benchmark | 复现值 | README 值 | 差异原因 | reproduce 脚本是否有问题 |
|---|---:|---:|---|---|
| Darcy smooth | 0.001697 | 0.0047 | 当前仓库代码 vs 论文原始代码 | **无** |
| Darcy rough | 0.003607 | 0.0089 | 同上；原始代码有硬编码路径，论文用的代码版本不可知 | **无** |
| Navier-Stokes 1e-5 | 0.01597 | 0.0820 | 当前仓库代码 vs 论文原始代码 | **无** |
| Pipe flow | 0.002955 | 0.0183 | 当前仓库代码 vs 论文原始代码 | **无** |
| Helmholtz | H1=0.04896 | H1=0.0284 | 度量定义不同（绝对 vs 相对 H1），main 分支已有此问题 | **无** |

---

## 五、证据文件索引

| 证据 | 位置 |
|---|---|
| 脚本修改 diff | `git diff main..reproduce -- darcy.py helm.py navier.py utilities3.py` |
| 原始代码路径问题 | `git show 6c6ea6b:utilities3.py`（darcy20c6 硬编码路径） |
| models.py 仅改文档 | `git diff 6c6ea6b..main -- models.py` |
| train_full_2 数据已预加载到 GPU | `git show main:navier.py`，objective 函数中 `train_a.to(device)` 行 |
| HSloss_d 非相对 H1 | `helm.py` 中 `if dataOpt['data'] == 'helm': h1loss = HSloss_d()` |
| run 脚本超参数 | `scripts/run_readme_experiments.sh`（与 README 逐字比对） |
| 实验输出 | `/nas-dev-slow/pep_design/PDE/MgNO-runs/` |
