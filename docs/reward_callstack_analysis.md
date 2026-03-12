# Reward Computation Call Stack Analysis

## Overview

Traces the reward computation flow in verl, covering all three paths:
legacy (FSDP/Megatron), RewardLoop (Ray actors), and rule-based (rank 0).

---

## Training Loop Entry

**File:** `verl/trainer/ppo/ray_trainer.py`

`fit()` (~line 1353) drives the main training loop. Reward computation is
triggered after rollout generation via `_compute_or_extract_reward(batch)`.

---

## Three Reward Paths

### 1. Legacy FSDP/Megatron Path (`use_reward_loop=False`, RM enabled)

RM forward pass runs on each DP rank locally. Rule-based reward function
runs on rank 0 only.

```
ray_trainer.fit()
  -> _compute_or_extract_reward(batch)
    -> reward.compute_reward(data, reward_fn)
      -> reward_fn(data, return_dict=True)  # AbstractRewardManager.__call__
        -> per-sample loop:
             self.compute_score(data_source, solution_str, ground_truth, extra_info)
```

### 2. RewardLoop Path (`use_reward_loop=True`)

Batch chunked across N `RewardLoopWorker` Ray actors. Rewards computed in
parallel. Results reassembled on the controller.

```
ray_trainer.fit()
  -> self.reward_loop_manager.compute_rm_score(batch)
    -> data.chunk(N) -> workers
      -> RewardLoopWorker.compute_score_batch(chunk)
        -> for item in chunk:
             self.reward_loop.run_single(item)
               -> compute_score(data_source, solution_str, ground_truth, extra_info)
    -> ray.get() -> reassemble rm_scores tensor
```

### 3. Rule-Based Path (RM `enable=False`)

`compute_score` runs on the trainer process (rank 0). Can also run async
via `compute_reward_async` Ray task.

```
ray_trainer.fit()
  -> _compute_or_extract_reward(batch)
    -> reward.compute_reward(data, reward_fn)
      -> reward_fn(data, return_dict=True)
```

---

## Key Files

| Component | Path |
|---|---|
| Training loop | `verl/trainer/ppo/ray_trainer.py` |
| Reward dispatch | `verl/trainer/ppo/reward.py` (`load_reward_manager`, `compute_reward`) |
| RewardLoopManager | `verl/experimental/reward_loop/reward_loop.py` (lines 227-305) |
| RewardLoopWorker | `verl/experimental/reward_loop/reward_loop.py` (lines 39-224) |
| DAPO reward manager (legacy) | `verl/workers/reward_manager/dapo.py` |
| DAPO reward manager (reward_loop) | `verl/experimental/reward_loop/reward_manager/dapo.py` |
| Reward manager registry | `verl/workers/reward_manager/registry.py` |

---

## Reward Distribution Details

### RewardLoopManager.compute_rm_score

Located at `verl/experimental/reward_loop/reward_loop.py` lines 227-305.

1. Input batch is chunked into N pieces (one per `RewardLoopWorker` actor)
2. Each worker calls `run_single()` on every item in its chunk asynchronously
3. Controller collects results via `ray.get()`
4. Scores reassembled into a single `rm_scores` tensor

### RewardLoopWorker.run_single

Located at `verl/experimental/reward_loop/reward_loop.py` lines 39-224.

Each worker:
- Extracts `data_source`, `solution_str`, `ground_truth`, `extra_info` from item
- Calls `compute_score(data_source, solution_str, ground_truth, extra_info)`
- Returns scalar reward

### Legacy compute_reward

Located at `verl/trainer/ppo/reward.py`.

- `load_reward_manager()` resolves the reward manager class from registry
- `compute_reward(data, reward_fn)` calls `reward_fn(data, return_dict=True)`
- `reward_fn` is `AbstractRewardManager.__call__`, which loops over samples

---

## FAQ

**Q: Is reward graded by a single controller?**

Depends on the path:
- **RewardLoop:** distributed across `RewardLoopWorker` Ray actors, then
  aggregated on the controller.
- **Legacy:** `compute_reward` runs on the controller (rank 0), but the RM
  forward pass itself is distributed across DP ranks.

**Q: Do all DP ranks send responses back to rank 0?**

Yes. Training data is gathered on the controller before reward computation,
then scattered back to DP ranks after scoring.

**Q: Where does async reward happen?**

`compute_reward_async` is a Ray remote task used in the rule-based path.
It allows reward computation to overlap with other pipeline stages.

---

## Notes

- The `compute_score` signature is consistent across all paths:
  `compute_score(data_source, solution_str, ground_truth, extra_info)`
- Registry at `verl/workers/reward_manager/registry.py` maps string names
  to reward manager classes
- RewardLoop workers are long-lived Ray actors, not ephemeral tasks
