# Findings — AFS root_squash behavior on QuarkFS CSI

**Rounds**: Round 1 (2026-04-16 06:00–06:10 UTC), Round 2 (2026-04-16 10:52 UTC — re-verify after admin said "fixed")
**Cluster**: vcluster on SenseCore k8s, QuarkFS (quarkfs-sc / csi.quarkfs.com)

---

## Round 2 (2026-04-16 10:52 UTC) — ADMIN FIX NOT EFFECTIVE

Admin reported the squash issue as fixed. Re-ran the exact same 3-PVC probe
methodology. Result: **no change**. Same squash, same mount options, same
behavior across all three annotation values.

| PVC | Pod uid | File created by pod — UID on PFS | FUSE mount options |
|---|---|---|---|
| `squash-test-false` | 0 (root) | **10000/UNKNOWN** ❌ | `rw,relatime,user_id=0,group_id=0,default_permissions,allow_other` |
| `squash-test-true` | 0 (root) | **10000/UNKNOWN** ❌ | `rw,relatime,user_id=0,group_id=0,allow_other` |
| `squash-test-default` | 0 (root) | **10000/UNKNOWN** ❌ | `rw,relatime,user_id=0,group_id=0,default_permissions,allow_other` |

Logs: `logs/probe-{false,true,default}.log` (round 2) and `logs/round1-20260416/` (round 1).

Comparison — ***identical*** to round 1 along every axis:
1. All three pods ran as `uid=0(root) gid=0(root)`.
2. Every file/dir created on AFS ended up owned by `10000/UNKNOWN` on the host.
3. The only annotation-dependent difference remains the `default_permissions`
   FUSE flag (absent when `afs.root_squash: "true"`, present otherwise).
4. Pre-existing root-owned dirs (`asr/`, `io_test/`, `kk/`, `lichang93/`) are
   still root-owned on the host view, confirming the filesystem itself can
   hold root-owned files — they just aren't writable as root from our pods.
5. Round 1's probe files (from 06:02–06:05 UTC) are still visible in
   `/mnt/pfs/` — still owned by 10000.

**Conclusion of round 2**: whatever admin did, the effect on this PVC +
StorageClass + `jdafs` secret combination is nil. Either the admin fix
targeted a different resource (wrong PVC, wrong namespace, wrong CSI driver),
or the intended change didn't propagate to the running CSI pods, or the fix
assumes a client-side mount option that this CSI driver doesn't honor.

### What to report back to admin (round 2)

> 李老师 重新测试了 — 问题没解决。
>
> 复现步骤: 新建 3 个 PVC (annotation `afs.root_squash` 分别为 `true` / `false` / 未设置)，在每个 PVC 上跑一个 uid=0 root 的 pod，创建文件并看 host 上的 owner。三个 PVC 的结果都一样: **文件 owner = 10000 / UNKNOWN**。跟上次 4月16日早上6点测试完全一致。
>
> 之前的诊断结论还成立: 这个 squash 是 **server-side** 由 `jdafs` access key 决定的，不是 CSI mount options 能覆盖的。想让 pod 作为 root 写 AFS，需要:
> - (a) 换一个 access key，它对应 server 侧的 uid=0 身份；或
> - (b) 把 `jdafs` 现有 access key 在 server 侧映射到 uid=0；或
> - (c) 在 quarkfs server 的这个 AFS 卷上显式开一个类似 NFS `no_root_squash` 的 per-volume option（如果 QuarkFS 支持）。
>
> 现在的 workaround 是每次在 host 上写了文件给 pod 读之前跑 `chmod -R o+rX`，但不是长久之计。
>
> 完整测试日志: `/mnt/public/lichang93/st_verl_dockerfile/squash_test/logs/`

### Round 2 cleanup

All round-2 test objects deleted (`kubectl delete pvc squash-test-* pod -l squashtest=true`). Confirmed clean.
The round-1 probe files/dirs left on `/mnt/pfs/` are still there (owned by 10000, manually removable).

### Round 2b — direct test against the (re-created) production PVC `pvc-jdwzpnv`

User caught a blind spot in Round 2: I had only tested freshly-created
*squash-test-\** PVCs, not the actual production PVC `pvc-jdwzpnv`. On
closer inspection, `pvc-jdwzpnv` had been **recreated by the admin at
2026-04-16 08:19 UTC** (new underlying PV `pvc-e100c838-…`, ~2.5h old at
test time). So the admin's fix attempt was specifically to recreate this
PVC — which Round 2 never tested.

Re-ran the probe against `pvc-jdwzpnv` directly (pod mounts the full PVC at
`/mnt/pfs`, writes a file under `lichang93/st_verl_dockerfile/squash_test/`
so the host can see it at `/mnt/public/lichang93/…`):

```
INSIDE POD (at 2026-04-16 10:57 UTC)
  id       → uid=0(root) gid=0(root)
  mount    → quarkfs_client … user_id=0,group_id=0,default_permissions,allow_other
  stat     → Uid: (10000/UNKNOWN) Gid: (10000/UNKNOWN)

HOST
  ls -la prod-probe-105731.txt
  → -rw-r--r-- 1 10000 10000 48 Apr 16 10:57
```

**Result: identical squash behavior**. Recreating the PVC did not change
anything — because the PVC points at the same `afs.endpoint` with the same
`afs.secretName: jdafs`, and the squash is determined server-side by the
access key's user mapping, not by the PVC lifecycle.

### Updated takeaway for admin

> 老师 重新跑了一遍，这次直接针对您今天早上 08:19 UTC 重建的 `pvc-jdwzpnv` 测试 (而不是我之前新建的 squash-test-* PVC) —— 行为一样，pod 里 root 写文件在 host 上还是 10000。
>
> 重建 PVC 本身没用，因为新 PVC 还是:
> - 指向同一个 `afs.endpoint` (019c70e4-68da-…)
> - 用同一个 `afs.secretName: jdafs`
> - 同样的 FUSE mount option (`user_id=0,group_id=0,default_permissions,allow_other`)
>
> Squash 不是 CSI / PVC 层决定的，是 quarkfs 服务端根据 `jdafs` 这个 access key 的用户映射决定的。只要这个 key 还绑在 uid 10000 上，不管 PVC 怎么重建都没用。需要在服务端做 (a) 换 key、(b) 把 key 映射到 uid 0、或 (c) 给这个卷开 per-volume no_root_squash 之一。
>
> 完整证据: `/mnt/public/lichang93/st_verl_dockerfile/squash_test/findings.md` §Round 2b

---

## Round 1 (2026-04-16 06:00 UTC) — original findings, preserved below for history

## Executive summary

**The `afs.root_squash: "false"` PVC annotation does NOT prevent root-squashing on the `csi.quarkfs.com` driver as currently deployed on this cluster.** The squash is enforced **server-side** by the AccessKey/SecretKey credentials in the `jdafs` secret — the PVC annotation is functionally a no-op for squash.

**The admin's suggestion ("显式设置为 false") does not solve the problem** because the annotation is already set to `false` on the existing PVC, and the test confirms that PVCs freshly created with `false`, `true`, or **no** annotation all behave identically (same mount, same squash).

---

## Test setup

Three PVCs created, all on the same StorageClass (`quarkfs-sc`), same
`afs.endpoint`, same `afs.secretName: jdafs`:

| PVC | `afs.root_squash` annotation |
|---|---|
| `squash-test-true` | `"true"` |
| `squash-test-false` | `"false"` |
| `squash-test-default` | (not set — control) |

A probe pod (running uid=0 inside container, based on our standard
`myverl:ncr2602_vllm012.dev` image) was deployed against each PVC. The pod:
- Printed its process identity (`id`).
- Inspected the FUSE mount options (`mount | grep /mnt`).
- Created a file and a dir, then inspected their owner UIDs.
- Tried to read an existing root-owned file under `lichang93/`.

Full logs: `squash_test/logs/probe-{true,false,default}.log`.

## Observations

### 1. All three PVCs mount the same AFS directory

The probe pods for `false`, `true`, and `default` each saw each other's
earlier writes:
```
drwxr-xr-x  2 10000 10000 4096 Apr 16 06:02 probe-dir-squash-test-false-060208
drwxr-xr-x  2 10000 10000 4096 Apr 16 06:03 probe-dir-squash-test-true-060340
drwxr-xr-x  2 10000 10000 4096 Apr 16 06:05 probe-dir-squash-test-default-060547
```
The three PVCs are distinct k8s objects (different UIDs, different bind
events), but at the data level they are all mapped to the same AFS root
directory. The PVC annotation `afs.endpoint` (which points at a specific
AFS volume GUID) is the only thing that selects which AFS volume to mount;
the CSI driver doesn't carve out a per-PVC subdirectory.

### 2. Identical squash behavior across all three PVCs

Inside every probe pod `id` reports:
```
uid=0(root) gid=0(root) groups=0(root)
```

But every file the pod creates shows:
```
Access: (0644/-rw-r--r--)  Uid: (10000/ UNKNOWN)   Gid: (10000/ UNKNOWN)
```

This is identical across `false`, `true`, and `default` PVCs.

### 3. The only CSI-visible difference between annotations is one FUSE flag

Mount option strings observed:

| PVC | Mount options |
|---|---|
| `false` | `rw,relatime,user_id=0,group_id=0,default_permissions,allow_other` |
| `default` | `rw,relatime,user_id=0,group_id=0,default_permissions,allow_other` |
| `true` | `rw,relatime,user_id=0,group_id=0,allow_other` (no `default_permissions`) |

So the annotation does affect exactly **one** FUSE option: whether
`default_permissions` is passed to the kernel. That flag controls whether
the **kernel** does a local perm check before delegating to the FUSE
daemon; it is unrelated to who owns files that the daemon writes on the
server.

### 4. The squash is enforced server-side

The pod's uid=0 is translated to uid 10000 **at write time** regardless of
mount options. This can only be a server-side decision: the quarkfs MDS
sees the request authenticated by the `jdafs` access key, looks up the
user that key is mapped to, and assigns that uid on create.

Evidence that server-side identity is the governing factor:
- The `jdafs` secret contains `accessKey` and `secretKey` (decoded values:
  `019C711ABD1C722A8F89FE4D234A0E05` and `019C711ABD1C721D977CD7FBDF891447`).
- Files created by any pod using this secret come out owned by uid 10000.
- Pre-existing root-owned directories (`asr/`, `io_test/`, `kk/`, `lichang93/`)
  must have been created by a different AFS identity (probably an admin
  key, or a CLI mount outside k8s that authenticated as root).
- The `io_bench/` directory is owned by `10000`, consistent with another
  tenant using the same kind of per-tenant access key that maps to 10000.

### 5. Read access to root-owned files is not universally denied

The probe's read test picked a root-owned file in `lichang93/` and
successfully read it (exit code 0):
```
Attempting to read: /mnt/pfs/lichang93/A2mWGebZRmePSKfjeODhBA.tar.gz
(read test exit code: 0)
```

This confirms: read failures depend on file mode, not on "pod is squashed."
Root-owned files with `o+r` perm are readable by the pod's effective
server-side identity (uid 10000). Files with `o=---` (like the
`gpu_burn.py` we saw in voltest Bug #1 with mode 750) are **not**
readable.

The voltest fix (`chmod -R o+rX`) is a workaround at the perm level; it
works because uid 10000 falls into the "other" category and needs "other
read" perms. The real fix (making the pod appear as root at the server
level) requires admin action on the server side.

## Why the admin's suggestion didn't fix it

> 李老师 可以把这个显式设置为 false 看看
> afs.root_squash: "false"

The existing production PVC `pvc-jdwzpnv` has had this annotation
set to `false` the whole time (confirmed via `kubectl get pvc pvc-jdwzpnv
-o jsonpath='{.metadata.annotations}'`). Our new test PVCs confirm that
regardless of annotation value, root is squashed to uid 10000.

The annotation exists in the CSI driver's vocabulary, but on this cluster
the only thing it changes is the FUSE `default_permissions` flag, which
is unrelated to server-side identity mapping.

## What to ask the admin

1. **Confirm**: the `jdafs` access key is mapped to user **uid 10000** on the
   quarkfs server. Is that intentional, and is there an admin key that
   maps to uid 0?
2. **Request one of**:
   - (a) A **new secret** with an access key that authenticates as uid 0
     (or equivalent "root on AFS" identity), so our pods can present as
     root to the filesystem and create/read root-owned files naturally.
   - (b) **Server-side policy change**: map the existing `jdafs` access
     key to uid 0 / root-equivalent privileges.
   - (c) Clarify whether QuarkFS supports the equivalent of NFS's
     `no_root_squash` export option on a per-share basis, and if so,
     enable it on our AFS volume.

Any of (a), (b), or (c) would let us write to PFS as root from pods
without the `chmod -R o+rX` workaround. Without one of these, we have to
keep running `chmod -R o+rX` on any code/data the pods need to read.

3. **Clarify**: what does the `afs.root_squash` annotation actually do on
   this driver? Our evidence is it only toggles the FUSE `default_permissions`
   flag, not server-side squash. Worth documenting in the internal
   quarkfs docs (if not already).

## Recommended follow-up

Until admin resolves the server-side identity issue, document the
workaround in our install guide:

- See `k8s_kuberay_kueue_setup/install_from_scratch.md` Appendix A (already
  present) — continue using `chmod -R o+rX` after creating files on PFS.
- Consider moving the workaround into the image itself: have the container
  entrypoint do `chmod -R o+rX /root/myCodeLab/host/` on startup. Not
  ideal (touches timestamps, slow on large dirs) but fully pod-local.
- **Do NOT** create new PVCs expecting different squash behavior — proven
  not to work on this driver.

## Cleanup

All test objects deleted at the end of this run (see `cleanup.sh` or run
`kubectl delete pvc squash-test-true squash-test-false squash-test-default`
and `kubectl delete pod -l squashtest=true`). Probe files left on AFS under
`/mnt/pfs/` can be removed manually if desired (they are owned by 10000 so
any pod using the `jdafs` secret can delete them).
