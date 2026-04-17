# How to Reproduce voltest Bug #1 (PFS EACCES)

**Verified end-to-end**: 2026-04-17 on `pvc-jdwzpnv` / `quarkfs-sc` / KubeRay.

## The bug

voltest/dev_notes.md:28–39 recorded: during the voltest RayJob, the training process opening `/root/myCodeLab/host/voltest/gpu_burn.py` failed with:
```
python3: can't open file '/root/myCodeLab/host/voltest/gpu_burn.py': [Errno 13] Permission denied
```
The fix applied at the time: `chmod -R o+rX /mnt/public/lichang93/st_verl_dockerfile/voltest/`.

We reproduce the identical error **without any artificial setup** — only the natural file-creation path that every host-side workflow does.

## Minimal natural reproduction

No `chown`, no explicit `chmod 640`. Everything below happens organically:

| Step | Where | Action | Resulting state |
|---|---|---|---|
| 1 | HOST | `echo 'print("hi")' > victim.py`  *(or `vi victim.py`)* | `-rw-r----- root root` (mode 640, owner 0, **via host `umask 0027`**) |
| 2 | HOST | `chmod +x victim.py` | `-rwxr-x--- root root` (mode **750**, **same bits voltest's `gpu_burn.py` had**) |
| 3 | POD  | training/actor process opens the script | `[Errno 13] Permission denied` |

That's it. No setup trick: mode 750 is what the default umask + `chmod +x` naturally produces on this host.

## Why it fails

Three layers together:

1. **Host `umask 0027`** — set by `/etc/profile` and `/etc/bash.bashrc`. Any host-created file starts as mode `640` (no "other" read bit). A subsequent `chmod +x` (needed to make a script executable) bumps it to `750` — still no "other" read bit, because `+x` only sets bits that already had read.

2. **File owner stays root on server when host writes** (tested 2026-04-17). So the file is `root:root 750` on PFS.

3. **Non-root processes in the pod lose `CAP_DAC_OVERRIDE`**. A plain root process inside the pod has CAP_DAC_OVERRIDE and bypasses all DAC checks — it would read the file fine. But many real processes in a pod run at a reduced privilege: `su -s bash nobody`, `setpriv --reuid=X`, python process that called `os.setuid()`, Ray actors that drop privileges, `ccr-jd` tenant identity, etc. Any non-root reader hits a standard UNIX DAC check against the file:
   - process uid (e.g. 65534) ≠ file owner uid (0) → not owner
   - process gid (65534) ≠ file group gid (0) → not in group
   - falls to **other** → mode 750 "other" = `---` → EACCES

## Why the voltest fix (`chmod -R o+rX`) works

It adds "other read" to files (via `o+r`) and "other execute" (traverse) to directories and already-executable files (via `o+X`). After the fix, mode becomes `755` → "other" now has `r-x` → the DAC check passes even for non-root readers. Original uid mismatch is unchanged — the fix simply opens the "other" perm class.

## Minimal test infrastructure

`yaml/rayjob-repro.yaml` — a 1-head, 0-worker RayJob with the production PVC (`pvc-jdwzpnv`) mounted at `/root/myCodeLab/host` via the same subPath (`lichang93/st_verl_dockerfile`) the real training uses. Entrypoint is `sleep 7200` so the head pod stays alive for the manual reproduction. No GPU requested so it schedules anywhere.

**Note on the kuberay auto-submitter pod**: gets `Pending` under volcano on this cluster because volcano binds it into the podgroup but can't schedule it. We don't use the submitter — the head pod alone is enough for reproduction. The identity that matters (root vs non-root) is about the process opening the file, not which RayJob pod.

**Note on `/root` inside the pod**: by default it's mode `700`, so even with the file's "other" bits set correctly, a non-root test process can't traverse from `/` down to `/root/myCodeLab/host/...`. For reproduction convenience we `chmod 755 /root` inside the pod (container-local, ephemeral). In the original voltest scenario, the reader process was launched with a working directory already inside the mount (Ray actors start with cwd from their runtime env), so it never path-walked through `/root`.

## Natural trigger outside tests

Any workflow like this hits the bug:

1. Write a script on the host (any editor / tool respecting umask).
2. `chmod +x script.py` so it's runnable.
3. Submit a RayJob or k8s Job that exec's the script under a non-root user (including inside a container that does `USER ray` or similar, or any `su` / `setpriv` in a wrapper script).

The bug surfaces because step 1+2 naturally produces a mode-750 script that non-root readers can't open.

## Preventive options

### Fix A — one-shot retro-fix after host writes (voltest pattern)

Useful to clean up files that already exist with mode 640/750 on PFS.

```bash
chmod -R o+rX /mnt/public/lichang93/st_verl_dockerfile/<dir>/
```

- `o+r` adds "other read" to all files.
- `o+X` adds "other execute" only to directories and already-executable files (so it makes directories traversable and keeps `chmod +x`-ed scripts runnable by non-root, without accidentally making text files executable).

This is a **post-hoc** fix. You have to remember to run it after each batch of host writes.

### Fix B — change host `umask` so new files are born readable (preventive)

Host default is `umask 0027` (from `/etc/profile:28` and `/etc/bash.bashrc:72`) → new files are `640` → `chmod +x` → `750` → bug.

Switching to `umask 0022` → new files are `644` → `chmod +x` → `755` → **other-read is always present, no bug**.

Three scopes for the change:

**B.1 — per-shell (lowest blast radius, for a one-off):**
```bash
umask 022              # valid only in the current shell
umask                  # verify: 0022
touch /tmp/x && ls -la /tmp/x    # -rw-r--r--
```
Dies when the shell exits.

**B.2 — per-user (recommended — scoped to root's interactive sessions):**
```bash
echo 'umask 022' >> ~/.bashrc      # or ~/.profile for login shells
# takes effect in every NEW shell started after this
```
**Caveat**: a change to `~/.bashrc` does NOT affect already-running processes. Notably, if Claude Code (or any tool) is running, its child `Bash` tool still inherits the old umask until you restart. To apply to the live Claude Code session manually:
```bash
umask 022              # run inside the current Claude Code session's Bash tool
```
For a clean state, restart Claude Code after the `~/.bashrc` change.

**B.3 — system-wide (all users on this host):**
```bash
sudo sed -i 's/^umask 027/umask 022/' /etc/profile /etc/bash.bashrc
```
Blast radius: every user on b32 picks up the new default in new logins. Double-check files that intentionally need restrictive modes:
- `~/.ssh/` — SSH refuses to use 644 private keys. Spot check: `ls -la ~/.ssh/id_*`. If they became 644, restore with `chmod 600 ~/.ssh/id_*`.
- `kubeconfig` — `kubectl` warns on wide-readable configs. Check `ls -la ~/.kube/config`.
- `/etc` configs written post-change — usually installers set explicit modes, but audit anything written during package installs after the switch.

### Fix C — pod-side: keep readers as root (not recommended)

If every process that ever opens PFS files is uid 0 with `CAP_DAC_OVERRIDE`, the DAC check is bypassed and mode doesn't matter. But this forces every workload (Ray actors, sidecars, wrapper scripts) to stay root, which contradicts k8s security best practice and doesn't help when something in the chain does `setpriv` / `su` anyway. Mentioned for completeness; Fix B is the real answer.

### Recommendation

**Fix B.2** (`umask 022` in `~/.bashrc`) + **Fix A** once (`chmod -R o+rX` on existing PFS dirs). After that, new files created by Claude Code / vi / editors on this host will be `644`/`755` by default and the bug won't surface from normal workflows.

## Files

- `how_to_reproduce_voltest_bug1.md` — this doc
- `yaml/rayjob-repro.yaml` — minimal RayJob used for the test
- `findings.md` — earlier investigation (annotation experiments, still valid for server-side squash story)
