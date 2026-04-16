# squash_test — Root-Squash Diagnosis Plan

> Goal: figure out *why* files written by root on the host come out owned by
> UID 10000 inside k8s pods on our QuarkFS PVC, even though the admin's
> suggested annotation `afs.root_squash: "false"` is already set.
>
> The fix is **not** `chown` or `chmod` — we want the CSI+filesystem to
> present root as root.

---

## What we already know

From probing (16 Apr 2026 05:55 UTC):

```bash
kubectl get pvc pvc-jdwzpnv -o jsonpath='{.metadata.annotations}'
```
```
{
  "afs.endpoint":     "afs://019c70e4-68da-73af-9a70-e47e0c9b5ac6.mds.cluster1.cn-sh-01e.sensecore.cn",
  "afs.root_squash":  "false",                                 ← already set
  "afs.secretName":   "jdafs",
  "volume.beta.kubernetes.io/storage-provisioner": "csi.quarkfs.com"
}
```

Yet during the 2026-04-15 voltest run, every file created by root in a pod
mounted at `/root/myCodeLab/host` came out owned by UID 10000 on the host PFS:
```
-rw-r--r-- 1 10000 10000 18229 Apr 15 15:43 voltest-3.log
```

So the annotation is **present but not effective**. Admin's suggestion to
"explicitly set `afs.root_squash: "false"`" will only help if our annotation
was somehow being interpreted differently, or if a brand-new PVC created
with the annotation at creation time behaves differently from an
already-existing PVC with the same annotation added later.

Other facts:
- StorageClass `quarkfs-sc`: provisioner `csi.quarkfs.com`, **no parameters** set
  at SC level (so everything flows from PVC annotations).
- Secret `jdafs` exists and is referenced by the PVC annotation `afs.secretName`.
- PV has `flexVolume.driver: fake` — this is a vCluster Loft "fake-pv"
  wrapper; the real mount happens at the host cluster level.
- The `csi.quarkfs.com` CSI driver is **not visible in this vcluster** (no
  pods in any visible namespace) — it runs on the host cluster.
- No docs publicly available for this CSI driver (internal Aliyun/SenseCore).

## Hypotheses (ordered by likelihood)

| # | Hypothesis | How to test |
|---|---|---|
| H1 | Annotation is read by the CSI driver **only at PVC creation time** and our existing PVC had a different/missing value when first provisioned — the later-added `false` value was not propagated to the active mount | Create a brand-new PVC with `afs.root_squash: "false"` at birth and see if it behaves differently |
| H2 | Annotation is read correctly but overridden by a cluster-wide or server-side default on the AFS filesystem | Ask admin if there's a server-side config; compare behavior across multiple new PVCs with different annotation values |
| H3 | The vcluster-Loft fake-pv wrapper drops or mangles annotations when forwarding to the host CSI driver | Create a test where we can compare behavior inside vs outside the vcluster (we cannot easily exit the vcluster, so this is hard) |
| H4 | The annotation key is wrong — maybe the driver expects e.g. `csi.quarkfs.com/root-squash` or a different spelling | Submit a PVC with multiple annotation variants, see which (if any) actually propagates |
| H5 | The annotation has to be a StorageClass parameter, not a PVC annotation | Ask admin to create a second StorageClass with `parameters: {root_squash: "false"}` or similar, provision a PVC from it |
| H6 | Setting is honoured but only when pod runs with `fsGroup` or specific `securityContext` that matches | Run test pod with different `securityContext` values (root user, fsGroup=0, etc.) |

## Files we will create

```
squash_test/
├── plan.md                   This file
├── dev_notes.md              Live log of the investigation
├── yaml/
│   ├── pvc-true.yaml         New PVC with afs.root_squash: "true"
│   ├── pvc-false.yaml        New PVC with afs.root_squash: "false"  (should be "correct")
│   ├── pvc-no-annotation.yaml   Control — no annotation at all
│   └── pod-probe.yaml        Probe pod that mounts a PVC and reports UID behavior
├── probe.sh                  Runs the probe pod against each PVC and captures results
└── logs/                     Per-PVC probe outputs
```

## What each probe test does (the "pod-probe" container will)

1. Print its own process UID/GID.
2. `ls -la /mnt/pfs/` before anything — see what files exist.
3. `touch /mnt/pfs/probe-from-pod-<pvcname>.txt` — create a file.
4. `stat /mnt/pfs/probe-from-pod-<pvcname>.txt` — see the resulting UID/GID from inside the pod.
5. `mount | grep /mnt/pfs` — see actual mount options.
6. Exit.

Then from the host:
7. `ls -la /mnt/public/lichang93/...<same path>` — see the host-visible UID/GID of the file we created.

**Interpretation grid**:

| Inside-pod UID (after touch) | Host-visible UID | Conclusion |
|---|---|---|
| 0 (root) | 0 (root) | **NO squash** — everything working as intended |
| 0 (root) | 10000 (or other non-zero) | **Squash at write time** — despite annotation |
| 10000 (or other) | 10000 | Pod was running as non-root already; can't tell if squash is in play |

## Phases

### Phase 0 — Probe current state (read-only)

```bash
# Verify PVC is bound, cluster idle
kubectl get pvc pvc-jdwzpnv
kubectl get pod -A | grep -v Completed | head
# Confirm no active workloads that could confuse the test
```

### Phase A — Create test PVCs and probe pod yamls

- `pvc-true.yaml` — annotation `afs.root_squash: "true"` (expected to squash)
- `pvc-false.yaml` — annotation `afs.root_squash: "false"` (admin says this should NOT squash)
- `pvc-no-annotation.yaml` — no `afs.root_squash` annotation at all (control, whatever the default is)
- `pod-probe.yaml` — reusable pod template with a `__PVC__` placeholder

All three new PVCs will reference the same StorageClass `quarkfs-sc` and the same `afs.secretName: jdafs`.

### Phase B — Run probe against each PVC sequentially

For each PVC:
1. `kubectl apply -f pvc-<variant>.yaml` — wait for Bound
2. Generate a pod manifest with `__PVC__` replaced by the PVC name
3. `kubectl apply -f` — wait for Running
4. `kubectl logs <pod>` — capture the inside-pod view
5. `ls -la /mnt/public/lichang93/squash_test/<pvcname>/` — host view
6. Record both UIDs in `dev_notes.md`
7. Delete the pod + PVC (or leave for inspection)

### Phase C — Analysis and recommendation

Fill out the interpretation grid for the 3 PVCs. Decide:
- If **pvc-false** shows root=root on both sides → fix is **create a new PVC, don't reuse the old one**. File a ticket to migrate `st_verl_dockerfile` subPath to a freshly-created PVC.
- If **all three** show squashed 10000 → the annotation does nothing; open a ticket with admin saying "annotation ineffective, please fix at CSI driver or StorageClass level".
- If results are mixed in a surprising way → document and ask admin for guidance with concrete evidence.

### Phase D — Document and close

Write a `findings.md` that includes:
- Exact test methodology
- Per-PVC UID observations
- Which hypothesis was correct
- The permanent fix (not a chown workaround)
- Updated guidance for `install_from_scratch.md` Appendix A

## Safety

- All new PVCs are isolated to `squash_test/` subdir; no impact on training PVCs.
- Probe pods request tiny resources (100m CPU, 128Mi RAM, no GPUs) → no scheduling contention.
- TTL on probe pod: 60 seconds of sleep then exit, so cluster stays clean.
- Dry-run each apply with `--dry-run=server` first where possible.

## Autonomous operation — no user input needed

User asleep. Will iterate Phase A → B → C → D and produce `findings.md` + an
updated `install_from_scratch.md` Appendix A without interrupting.

## References

- Kubernetes CSI driver concept: https://kubernetes-csi.github.io/docs/
- Internal: `voltest/dev_notes.md` (Bug #1 PFS root-squash observation)
- Internal: `k8s_kuberay_kueue_setup/install_from_scratch.md` Appendix A
