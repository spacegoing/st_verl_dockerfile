# k9s — quickstart

One page, 5 minutes, enough to start triaging real jobs.

## 0. Launch

```bash
k9s                                    # all namespaces, default context
k9s -n default                         # one namespace
k9s -c rayjob                          # jump straight to a resource view
k9s --readonly                         # disable delete/edit (prod safety)
```

On this cluster, the API server is reachable only when the proxy is
cleared. Mirror the `KP` alias:

```bash
alias K9='HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 k9s'
```

If colors look wrong: `export TERM=xterm-256color`.

## 1. The 6 keys you'll use all the time

| key           | what it does                                                             |
|---------------|--------------------------------------------------------------------------|
| `:`           | command mode — type a resource name and press enter                      |
| `/`           | filter the current view (regex by default)                               |
| `enter`       | drill in (rayjob → pods, pod → containers, …)                            |
| `esc`         | back out one level                                                       |
| `?`           | show all active key bindings for the current view                        |
| `:q`          | quit                                                                     |

## 2. Row actions — on the currently-highlighted row

| key             | action                                                 |
|-----------------|--------------------------------------------------------|
| `d`             | describe                                               |
| `y`             | view yaml                                              |
| `e`             | edit                                                   |
| `l`             | logs (inside, `f` follow, `w` wrap, `t` timestamps)    |
| `p`             | previous (crash-loop) logs                             |
| `s`             | shell in (pods)                                        |
| `f`             | port-forward (pods / services)                         |
| `ctrl-d`        | delete (prompts)                                       |
| `ctrl-k`        | kill (no prompt — careful)                             |
| `space`         | multi-select toggle                                    |

## 3. Command-mode recipes for this cluster

```
:rayjob                                              # all training runs
:rayjob /cbmd                                        # only the cbmd-* ones
:rayjob default training-type=40bra-md               # by label
:podgroup                                            # volcano gang scheduler
:pod ray.io/cluster=bra40-md-cbmd-v1-bfjpz-lcx5h     # pods of one ray cluster
:events                                              # cluster-wide events
:xray rayjob                                         # hierarchical view
:pulses                                              # cluster health
:ctx                                                 # switch kubecontext
:ns                                                  # switch namespace
```

## 4. Filter recipes (inside any view, press `/`)

| syntax                       | matches                                   |
|------------------------------|-------------------------------------------|
| `/cbmd-v2`                   | rows containing cbmd-v2                   |
| `/-l training-type=40bra-md` | rows with that label (server-side)        |
| `/-f ray`                    | fuzzy search over every column            |
| `/!SUCCEEDED`                | everything **except** SUCCEEDED rows      |

Esc clears the filter.

## 5. Daily workflow examples

### Morning: "what's running?"

```
:rayjob
/!SUCCEEDED       ← hides old terminal jobs
/!FAILED
```

### Mid-day: "why is my job stuck at Initializing?"

```
:podgroup                                        ← is it Inqueue?
:pod -l combo_id=cbmd-v2                         ← are pods Pending?
d   (on a Pending pod)                           ← events at the bottom explain why
```

### Training pod log tail

```
:rayjob
enter      (on your rayjob)                      ← drills into its pods
↓/↑        (select the head pod)
l          (logs) then f (follow)
```

### Delete all cbmd-v2 attempts

```
:rayjob
/cbmd-v2
space (on each row to multi-select) then ctrl-d
```

Or simpler:

```
:rayjob -l combo_id=cbmd-v2
ctrl-a (select all visible)   then ctrl-d
```

### Stuck / orphan PodGroups (when RayJob cascade-delete missed one)

```
:podgroup
/!Completed
ctrl-d  (on the orphaned row)
```

## 6. One-time config

Write `~/.config/k9s/config.yaml`:

```yaml
k9s:
  refreshRate: 5          # polite to a shared API server
  maxConnRetry: 5
  noExitOnCtrlC: true     # require :q to quit (prevents fat-finger exits)
  logger:
    tail: 400
    buffer: 5000
```

Write `~/.config/k9s/aliases.yaml` for our CRDs:

```yaml
aliases:
  rj: ray.io/v1/rayjobs
  rc: ray.io/v1/rayclusters
  pg: scheduling.volcano.sh/v1beta1/podgroups
  mymd: ray.io/v1/rayjobs default training-type=40bra-md
  mysd: ray.io/v1/rayjobs default training-type=40bra-sd
```

After saving you can type `:rj`, `:pg`, `:mymd` in command mode.

Write `~/.local/share/k9s/hotkeys.yaml` for one-key views:

```yaml
hotKeys:
  shift-0:
    shortCut: Shift-0
    description: all md RayJobs
    command: rayjobs default training-type=40bra-md
  shift-1:
    shortCut: Shift-1
    description: all sd RayJobs
    command: rayjobs default training-type=40bra-sd
  shift-2:
    shortCut: Shift-2
    description: PodGroups
    command: podgroups
```

Press `?` after saving to confirm your hotkeys show up in the help
panel.

## 7. Gotchas to save you 15 minutes each

- **Two different config dirs**: `~/.config/k9s` for `config.yaml` +
  `aliases.yaml`; `~/.local/share/k9s` for `hotkeys.yaml`, `plugins.yaml`,
  `screen-dumps/`. Run `k9s info` once to see both paths.
- **`ctrl-k` is kill with NO prompt.** Muscle memory says "ctrl-k clears
  the line" in bash. Don't.
- **Refresh-rate hits the API hard.** If the cluster is busy, bump to
  5–10 s.
- **Custom resources need CRD to be registered.** `:rayjob` only works
  because kuberay installed the CRD — nothing special on k9s's side.
- **Logs are truncated to `buffer`** (default 1000 lines). Raise it in
  `config.yaml` if you need more history.
- **Quit with `:q`, not `ctrl-c`**, or set `noExitOnCtrlC: true`.

## 8. Reference links

- Official home: <https://k9scli.io>
- Install topic (this was installed via the GitHub release, not the
  package manager — see `bisimpo/k9s_intro.md` §Install for reproduction):
  <https://k9scli.io/topics/install/>
- Commands reference: <https://k9scli.io/topics/commands/>
- Full primer in this repo: `bisimpo/k9s_intro.md`
