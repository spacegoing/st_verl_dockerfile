# voltest — Summary

> Verification that the KubeRay + Volcano gang-scheduling fix from
> `k8s_kuberay_kueue_setup/fix_kuberay_volcano_config_plan.md` (Phase 4)
> actually prevents the deadlock class observed in March Incidents A/B/C.

## What was tested

10 RayJobs submitted simultaneously, each requesting **8 B300 nodes × 8 GPUs = 64 GPUs**.
Total demand: 10 × 8 = **80 pods on a 31-node cluster** — gang-scheduling hard stress test.

- Ray app: [`gpu_burn.py`](gpu_burn.py) — dispatches 8 parallel `@ray.remote(num_gpus=8)` tasks, each running 5 min of `torch.matmul` on 8 GPUs.
- RayJob yaml: [`yaml/voltest-rayjob-template.yaml`](yaml/voltest-rayjob-template.yaml) — 1 head + 7 workers (minMember=8), whole-node resource sizing.
- Submission: [`launch.sh`](launch.sh).

## Headline result

**Gang scheduling works.** Volcano PodGroup enforces strict all-or-nothing placement.

| Check | Outcome |
|---|---|
| KubeRay auto-creates one PodGroup per RayJob | ✅ (10 PodGroups after launch) |
| PodGroup has correct `minMember=8` | ✅ |
| Only N jobs admit where N×8 ≤ free nodes | ✅ (3 jobs admitted, 24 pods on 24 nodes) |
| Remaining jobs held `Inqueue` with **zero** pod-to-node bindings | ✅ (56 pods Pending, all NODE=<none>) |
| When a running job finishes, next `Inqueue` PodGroup promotes atomically | ✅ (voltest-3 went 0/8 → 8/8 bound within 5s) |
| All 10 jobs eventually complete | ✅ (all `SUCCEEDED`) |
| Zero partial-placement deadlocks | ✅ |

Direct contrast with the March Incidents A/B/C: those deadlocked 2×16-node jobs on 33 nodes because KubeRay wasn't creating PodGroups. This test deliberately overcommits the cluster 2.6× and still succeeds cleanly.

## Timeline (actual)

| UTC | Event |
|---|---|
| 15:27:07 | voltest-0 submitted |
| 15:27:35 | voltest-1..9 submitted (9 more, within 1s) |
| 15:27:35 | 10 PodGroups exist: 3 Running (voltest-0,1,2), 7 Inqueue |
| 15:29:33 | voltest-0,1,2 each using 64/64 GPUs running GEMM |
| 15:34:54 | voltest-0 SUCCEEDED (manually deleted due to Bug #2 ttl=300) |
| 15:35:16 | voltest-3 promoted Inqueue → Running (gang, 8 pods bound atomically) |
| 15:36:14 | voltest-1, voltest-2 SUCCEEDED → auto-cleanup → voltest-4, voltest-5 promoted |
| 15:43:37 | voltest-3, voltest-4 SUCCEEDED → voltest-6, voltest-7 promoted |
| 15:43:57 | voltest-5 SUCCEEDED → voltest-8 promoted (third promotion) |
| 15:51:20 | voltest-6, voltest-7 SUCCEEDED → deleted |
| 15:51:41 | voltest-8 SUCCEEDED → deleted |
| 15:52:08 | voltest-9 promoted (final, was Inqueue for 24 min) |
| 15:59:04 | voltest-9 SUCCEEDED → deleted. **All 10 complete.** |

**Total wall clock: 31 min 57 sec (15:27:07 → 15:59:04).**

Throughput: 10 jobs × 64 GPUs × 5 min = 3200 GPU-minutes of work across 31 physical nodes. Cluster utilization during run: 3 × 8 = 24 of 31 nodes (~77%) — the 7-node gap is because 7 < 8 can't admit another 8-pod job.

## Bugs caught and fixed along the way

### Bug #1 — PFS root-squash permission denied
`gpu_burn.py` written with mode 750; pod UID 10000 couldn't read it. Fix: `chmod -R o+rX voltest/`. Documented as a recurring PFS gotcha.

### Bug #2 — `ttlSecondsAfterFinished` held nodes for 5 min after completion
Initial template had `ttlSecondsAfterFinished: 300`, so each SUCCEEDED RayJob kept its 8 pods Running for 5 more minutes, blocking promotion of next Inqueue PodGroup. Fix: reduced to `30` in the template and added auto-cleanup monitor that deletes terminal-state RayJobs immediately.

## Files produced

```
voltest/
├── plan.md                              Pre-execution plan (revised after user directive to scale up to 8-node jobs)
├── dev_notes.md                         Live execution log (phase-by-phase)
├── summary.md                           This file
├── gpu_burn.py                          Ray app (64-GPU GEMM burn)
├── launch.sh                            Submit N jobs from template
├── monitor.sh                           One-shot probe of state
├── collect_logs.sh                      Archive per-pod logs after run
├── yaml/
│   ├── voltest-rayjob-template.yaml     Parameterised RayJob (__JOBNAME__)
│   └── generated-voltest-{0..9}.yaml    Per-job instantiations
└── logs/
    ├── voltest-{0..9}.log               Per-job gpu_burn PFS logs
    ├── submit.log                        kubectl apply output
    ├── cleanup.log                       auto-cleanup events
    └── monitor-states.log                45s-interval state snapshots
```

## References

- KubeRay RayJob: https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html
- KubeRay + Volcano: https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html
- Volcano PodGroup: https://volcano.sh/en/docs/podgroup/
- Previous analysis: `k8s_kuberay_kueue_setup/kueue_vs_volcano_deep_analysis.md`
- Fix plan and dev notes: `k8s_kuberay_kueue_setup/fix_kuberay_volcano_config_plan.md` + `fix_kuberay_volcano_dev_notes.md`
