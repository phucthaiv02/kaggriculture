# Kaggriculture Dynamic Agent

Agent neural observation-conditioned cho Kaggriculture, được huấn luyện theo pipeline:

```text
Expert replays
    → Behavior Cloning
    → self-play recovery + AWR
    → league self-play + PPO
    → paired-seed promotion
    → package và runtime audit
```

Agent dùng residual CNN cho hai farm board, Transformer để kết hợp board/global/unit state và hai GRU autoregressive decoder cho unit actions và market orders. Legal masks cùng reservation state được dùng giống nhau trong inference, rollout và PPO.

## Chạy toàn bộ pipeline

### 1. Cài môi trường

Yêu cầu Python 3.11/3.12 và PyTorch phù hợp CUDA trên máy train:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[train,collect,dev]'
```

Kiểm tra H100/BF16:

```bash
python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.is_bf16_supported())"
```

Đăng nhập Kaggle trước khi thu expert replay:

```bash
kaggle auth login
```

Hoặc đặt token tại `~/.kaggle/access_token`.

### 2. Chỉnh cấu hình

Cấu hình cấp cao nằm tại [configs/pipeline.toml](configs/pipeline.toml). Mặc định pipeline chạy:

- top 10 teams × tối đa 200 replay/team;
- BC 30 epochs;
- 2 vòng AWR, mỗi vòng 500 seeds × hai vị trí;
- 10 vòng PPO, mỗi vòng 250 rollout seeds × hai vị trí trên 16 CPU workers;
- 50 paired promotion seeds × hai vị trí;
- một GPU; đặt `nproc_per_node = 8` để train BC/AWR bằng 8 H100.

Các hyperparameter chi tiết nằm tại:

- [configs/dataset.toml](configs/dataset.toml): sharding và relative value target;
- [configs/bc_h100.toml](configs/bc_h100.toml): Behavior Cloning;
- [configs/awr_h100.toml](configs/awr_h100.toml): AWR recovery;
- [configs/ppo_h100.toml](configs/ppo_h100.toml): PPO/GAE/KL;
- [configs/opponent_league.json](configs/opponent_league.json): opponent snapshot pool.

### 3. Chạy một lệnh

Từ thư mục gốc repo:

```bash
python scripts/run_full_pipeline.py --config configs/pipeline.toml
```

Lệnh trên tự động:

1. Thu expert replay nếu `data/raw/expert/manifest.json` chưa tồn tại.
2. Encode replay và tạo episode-split safetensors shards.
3. Train BC thành `checkpoints/agent_h100.pt`.
4. Đóng băng BC anchor tại `checkpoints/agent_bc_anchor.pt`.
5. Chạy các vòng stochastic self-play + AWR.
6. Khởi tạo `checkpoints/agent_best.pt` và opponent league.
7. Thu on-policy PPO trajectories, train candidate và đánh giá head-to-head.
8. Chỉ promote candidate vượt đủ win-rate, mean-margin và P10-margin gates.
9. Đóng gói `submissions/dynamic_agent.tar.gz` và audit archive.

Pipeline không ghi đè best checkpoint bằng PPO candidate chưa vượt promotion gate. Mỗi incumbent/candidate/promoted snapshot được giữ trong `checkpoints/` để có thể audit hoặc rollback.

### Tiếp tục một pipeline đã chạy

Nếu expert corpus và BC/AWR đã hoàn thành, tiếp tục từ PPO hiện có:

```bash
python scripts/run_full_pipeline.py \
  --skip-expert-collection \
  --skip-bc \
  --skip-awr
```

Chỉ đóng gói checkpoint best hiện tại:

```bash
python scripts/run_full_pipeline.py \
  --skip-expert-collection \
  --skip-bc \
  --skip-awr \
  --skip-ppo
```

Có thể thêm `--skip-package` khi chỉ muốn train.

## Pipeline học

### Behavior Cloning

Mỗi expert turn được encode thành:

- board `44 × 10 × 10` dạng `uint8`;
- global vector 70 chiều;
- tối đa 32 unit tokens;
- action operation/item/quantity targets;
- legal masks;
- relative value target.

Train/validation được split bằng hash của episode ID, nên turn của cùng một trận không rò rỉ sang cả hai split. Production value target là:

```text
clip((my_score - opponent_score) / 50000 + 0.25 × result, -5, 5)
```

Trong đó `result ∈ {-1, 0, 1}`. BC loss gồm action cross-entropy, value Smooth-L1 và illegal-probability penalties.

### Self-play recovery và AWR

BC policy được decode stochastic để đi vào các state không xuất hiện trong expert demonstrations. Expert và recovery replay được merge thành replay buffer tích lũy. AWR dùng:

```text
advantage = relative_final_target - V(state)
weight = min(exp(advantage / temperature), max_weight)
```

Pha này giúp policy học lại những trajectory recovery tốt trước khi bắt đầu PPO.

### League PPO

Mỗi PPO turn lưu:

- encoded observation;
- autoregressive action tokens;
- component activity masks;
- legal masks tại đúng thời điểm decode;
- joint old log-probability;
- old value, terminal reward và done.

Terminal reward dùng score margin cộng win/loss bonus. Trainer tính GAE rồi tối ưu clipped PPO objective cùng value loss, entropy bonus và KL penalty về frozen BC anchor:

```text
L = Lppo + value_coef × Lvalue - entropy_coef × entropy + kl_coef × KL(policy || BC)
```

Joint log-probability là tổng log-probability của các token thực sự tham gia action: unit op/item/quantity và market op/item/quantity, kể cả token `NONE` kết thúc market sequence.

Opponent được lấy có trọng số từ:

- policy hiện tại (`self`);
- built-in `starter`;
- các promoted historical snapshots.

Candidate được đánh giá trên cùng seeds ở cả player position 0 và 1. Mặc định chỉ promote nếu:

- win rate ≥ 52%;
- mean score margin ≥ 0;
- P10 score margin ≥ -25.000.

Các threshold nằm trong `configs/pipeline.toml` và nên tăng số promotion seeds trước khi chọn submission cuối.

## Chạy từng stage thủ công

### Expert collection và dataset

```bash
python scripts/collect_replays.py \
  --top-teams 10 \
  --episodes-per-team 200 \
  --max-discovery-queries 1000 \
  --output data/raw/expert

python scripts/build_dataset.py \
  --config configs/dataset.toml \
  --manifest data/raw/expert/manifest.json
```

### BC

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/train_bc.py --config configs/bc_h100.toml
```

### Một vòng AWR

```bash
python scripts/run_selfplay_iteration.py \
  --checkpoint checkpoints/agent_h100.pt \
  --iteration 1 \
  --base-manifest data/raw/expert/manifest.json \
  --seeds 500 \
  --opponent self \
  --opponent starter
```

### Một vòng PPO

```bash
python scripts/collect_ppo_rollouts.py \
  --checkpoint checkpoints/agent_best.pt \
  --league configs/opponent_league.json \
  --seeds 250 \
  --output data/ppo/iteration-001

python scripts/train_ppo.py \
  --config configs/ppo_h100.toml \
  --checkpoint checkpoints/agent_best.pt \
  --reference checkpoints/agent_bc_anchor.pt \
  --rollouts data/ppo/iteration-001 \
  --output checkpoints/candidates/ppo-iteration-001.pt

python scripts/promote_candidate.py \
  --candidate checkpoints/candidates/ppo-iteration-001.pt \
  --incumbent checkpoints/agent_best.pt \
  --best checkpoints/agent_best.pt \
  --league configs/opponent_league.json \
  --iteration 1
```

Khi chạy promotion thủ công, nên giữ một bản incumbent riêng nếu `--best` và `--incumbent` cùng đường dẫn. Full-pipeline orchestrator tự snapshot incumbent trước khi evaluation.

## Kiểm tra code

```bash
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
ruff check .
```

## Evaluation và đóng gói

Complete-game evaluation với built-in opponent:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/agent_best.pt \
  --opponent starter \
  --seed-count 100 \
  --save-worst-replay
```

Đóng gói và kiểm tra Kaggle runtime:

```bash
python scripts/package_submission.py \
  --checkpoint checkpoints/agent_best.pt \
  --output submissions/dynamic_agent.tar.gz

python scripts/audit_submission.py submissions/dynamic_agent.tar.gz
python scripts/test_runtime_image.py submissions/dynamic_agent.tar.gz
```

Submission archive bị từ chối nếu vượt 100 MiB. Runtime inference chỉ phụ thuộc Python standard library, NumPy và PyTorch có sẵn trong `gcr.io/kaggle-images/python:v163`.

## Artifact layout

```text
data/raw/expert/              expert replays và manifest
data/raw/selfplay/            AWR recovery replays
data/shards/                  BC/AWR safetensors
data/ppo/                     on-policy PPO trajectories
checkpoints/agent_bc_anchor.pt frozen BC KL anchor
checkpoints/agent_best.pt      checkpoint đã qua promotion
checkpoints/candidates/        PPO candidates
checkpoints/incumbents/        pre-update snapshots
checkpoints/league/            promoted historical opponents
runs/                          TensorBoard, evaluation và promotion reports
submissions/                   packaged Kaggle agent
```

Chi tiết model và quyết định thiết kế nằm tại [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); luật game nằm tại [docs/README.md](docs/README.md).
