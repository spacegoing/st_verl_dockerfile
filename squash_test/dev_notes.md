# squash_test — Dev Notes

> Live log. Latest at bottom.

## 2026-04-16 06:00 UTC — Phase 0 probe

Baseline discovery (see `plan.md` for full context):
- Current PVC `pvc-jdwzpnv` already has `afs.root_squash: "false"` but pods
  still write as UID 10000.
- StorageClass `quarkfs-sc`, provisioner `csi.quarkfs.com`, no SC parameters.
- vcluster Loft "fake-pv" wrapper on the PV.

Investigation proceeds via hypothesis H1 (annotation only honored at PVC
creation time): create brand-new PVCs with the correct annotation and
compare behavior.

## Phase A — yaml + probe pod authored (06:00–06:01 UTC)

Created:
- `yaml/pvc-true.yaml` — annotation `afs.root_squash: "true"`
- `yaml/pvc-false.yaml` — annotation `afs.root_squash: "false"`
- `yaml/pvc-default.yaml` — no `afs.root_squash` annotation (control)
- `yaml/pod-probe-template.yaml` — probe pod with `__PVC__` / `__PVC_SHORT__` placeholders; reports `id`, `mount`, file perms after touch, and tries reading an existing root-owned file
- `probe.sh` — applies a PVC variant, spawns the probe pod, waits for Running, captures logs
- `cleanup.sh` — removes all squash_test objects at end

## Phase B — probe runs (06:01–06:06 UTC)

Ran each probe sequentially. Full logs in `logs/probe-{true,false,default}.log`.

Key findings (see `findings.md` for full write-up):

1. **All three PVCs mount the same AFS directory.** Each probe pod saw
   the probe files created by the other two probes, confirming no
   per-PVC isolation at the data layer.

2. **Pods in all three cases run as `uid=0(root)`**, but every file they
   create comes out owned by **uid=10000 gid=10000**. Identical across
   all three annotation values.

3. **Annotation only affects one FUSE flag**:
   - `false` and `default` → `rw,relatime,user_id=0,group_id=0,default_permissions,allow_other`
   - `true` → same minus `default_permissions`

   The `default_permissions` flag enables kernel-level perm check caching
   and has nothing to do with server-side identity translation.

4. **Squash is enforced server-side.** The `jdafs` secret
   (accessKey=`019C...0E05`, secretKey=`019C...1447`) is mapped to a
   specific AFS user (uid 10000) on the QuarkFS MDS. Every write request
   authenticates with this key and the server assigns uid 10000 to the
   resulting inode.

5. **Read success depends on file mode**, not on squash. The probe read
   an existing root-owned `.tar.gz` in `lichang93/` successfully
   (because its mode was `o+r`). Voltest Bug #1 (`gpu_burn.py` mode 750,
   Permission denied) fits the same model: uid 10000 → "other" → perm
   check fails when mode is `o---`.

## Phase C — cleanup (06:06 UTC)

```
$ bash cleanup.sh
pod "squash-probe-default" deleted
pod "squash-probe-false" deleted
pod "squash-probe-true" deleted
persistentvolumeclaim "squash-test-true" deleted
persistentvolumeclaim "squash-test-false" deleted
persistentvolumeclaim "squash-test-default" deleted
(no squash-test PVCs remain)
```

## Phase D — findings documented

`findings.md` written with:
- Test evidence
- Root cause: server-side identity mapping via `jdafs` access key
- Three admin-resolvable options: (a) new secret with root-identity key,
  (b) server-side policy change to map existing key to root, (c) enable
  `no_root_squash` equivalent on AFS volume
- Why the admin's suggestion (set annotation to false) doesn't help:
  the annotation is already false AND newly created PVCs with it still
  squash

## Conclusion

**Manual `chmod -R o+rX`** is the only tenant-side workaround that works
without admin involvement. This is consistent with `install_from_scratch.md`
Appendix A. Until one of the three admin-side options is implemented, the
workaround remains.

**Action item**: take `findings.md` to the admin, request option (a), (b),
or (c), and point out that the current PVC annotation advice is a red
herring on this CSI driver.
