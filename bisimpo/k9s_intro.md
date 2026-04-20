# k9s — 0 to master

A terminal UI for Kubernetes. Live-updates every ~2 s, navigates any
resource with a two-key command, and wraps `kubectl describe / logs /
exec / delete / edit` behind single-letter hotkeys. Written by
[@derailed](https://github.com/derailed), BSD-licensed.

This document is the "read once, understand forever" primer. For the
30-second version see `k9s_quickstart.md` next to this file.

Source truth (quote-able): <https://k9scli.io>, <https://github.com/derailed/k9s>.
Installed version on this dev box: **v0.50.18** (Jan 2026).

---

## Install (from the official GitHub release)

The docs page at <https://k9scli.io/topics/install/> lists many
package-manager options (brew, scoop, choco, pacman, snap, curl|bash)
but they either require root + a package index we don't run, or route
through third parties. The most auditable path — and the one used on
this dev box — is the **tarball from the GitHub release page**.

### Pick the right asset

Releases live at <https://github.com/derailed/k9s/releases>. Each tag
ships ~24 assets: `.tar.gz` for every OS+arch (Linux/macOS/FreeBSD ×
amd64/arm64/armv7), `.deb`/`.rpm`/`.apk` for Linux package managers,
plus a `checksums.sha256` file and per-tarball SBOM JSON.

For our Linux amd64 box:

| asset                              | what it is                              |
|------------------------------------|-----------------------------------------|
| `k9s_Linux_amd64.tar.gz`           | the binary (note the capital `L`)       |
| `checksums.sha256`                 | SHA-256 for every asset                 |
| `k9s_Linux_amd64.tar.gz.sbom.json` | optional software bill of materials     |

### The exact commands I ran

```bash
VER=v0.50.18
BASE=https://github.com/derailed/k9s/releases/download/${VER}

# Proxy on during the download (GitHub needs it from this network)
HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY \
    curl -sSL -o /tmp/k9s.tar.gz     "${BASE}/k9s_Linux_amd64.tar.gz"
HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY \
    curl -sSL -o /tmp/checksums.sha256 "${BASE}/checksums.sha256"

# Verify BEFORE running the binary
cd /tmp && grep 'k9s_Linux_amd64.tar.gz$' checksums.sha256 | sha256sum -c -
# expected:  k9s_Linux_amd64.tar.gz: OK

# Extract, install, smoke-test
tar xzf /tmp/k9s.tar.gz -C /tmp
install -m 0755 /tmp/k9s /usr/local/bin/k9s
k9s version
```

Replace `$PROXY` with the dev cluster's proxy URL. On a cluster that
doesn't need a proxy, drop the `HTTPS_PROXY`/`HTTP_PROXY` prefixes.

### To install without root

`install -m 0755 k9s $HOME/.local/bin/` works if `~/.local/bin` is on
your `$PATH`. No other state is required — k9s is a single static Go
binary.

### Upgrade / downgrade

Repeat with a different `VER=`. k9s keeps its config in
`$XDG_CONFIG_HOME/k9s` and `$XDG_DATA_HOME/k9s` (see §2) so the binary
is safely swappable in place. Use the GitHub releases "latest" redirect
if you want the newest:

```bash
curl -sSL https://github.com/derailed/k9s/releases/latest/download/k9s_Linux_amd64.tar.gz -o /tmp/k9s.tar.gz
```

### Why not `go install`, `brew`, `snap`, etc.?

- `go install github.com/derailed/k9s@latest` — works if you have Go
  installed, but each release adds frontend assets that aren't always
  picked up cleanly by `go install`. The authors recommend pre-built
  binaries.
- `brew` / `snap` / `choco` / `scoop` — fine if you have the package
  manager, but they add a layer between you and the upstream release.
- Webinstall (`curl | bash`) — pipes to shell, opaque about version.

The tarball + checksum route lets you pin a specific version, verify
it, and audit the binary location. Zero surprises.

---

## 1. Mental model

k9s is *not* a replacement for `kubectl`. It's a **viewer with shortcuts**.
Everything you do in k9s is translated into a kubectl API call under the
hood. Three concepts cover 90% of usage:

1. **Views** — each screen is one view of one resource type (pods,
   rayjobs, podgroups, events, …). Views refresh on their own.
2. **Command mode** — hit `:` to switch views, like a Vim ex command.
   `:rayjob` → RayJob view. `:ns` → Namespace switcher.
3. **Hotkeys** — single letters on the currently-selected row trigger
   actions: `l` logs, `d` describe, `y` yaml, `ctrl-d` delete, `s` shell.

That's the whole UI. Everything else (XRay, Pulses, Plugins, Skins) is a
specialised view.

---

## 2. Configuration layout (the file tree k9s reads)

All files live under `$XDG_CONFIG_HOME/k9s` and `$XDG_DATA_HOME/k9s`.
On Linux without custom `$XDG_*` vars, that's:

```
~/.config/k9s/
├── config.yaml         # global settings (refreshRate, readOnly, …)
├── aliases.yaml        # custom :command aliases (see §6)
└── skins/              # color schemes

~/.local/share/k9s/
├── hotkeys.yaml        # keyboard shortcuts → commands (see §7)
├── plugins.yaml        # plugin definitions (see §8)
├── screen-dumps/       # where :screendump writes yaml snapshots
└── clusters/           # per-context overrides
    └── <cluster>/<context>/{config.yaml,aliases.yaml,hotkeys.yaml,…}
```

You can override the base location by exporting `K9S_CONFIG_DIR` before
launch. Handy if you want to keep k9s config in the repo rather than
`~`.

The global config.yaml uses these fields (from the official
`config.yaml` reference):

| field                    | default | meaning                                                          |
|--------------------------|---------|------------------------------------------------------------------|
| `refreshRate`            | 2       | UI poll interval (seconds). Bump to 5–10 on busy API servers.   |
| `liveViewAutoRefresh`    | false   | Refresh describe/yaml views while open                           |
| `apiServerTimeout`       | 120s    | single-call timeout                                              |
| `maxConnRetry`           | 15      | auto-reconnect attempts after API disconnect                     |
| `readOnly`               | false   | disables delete / edit / kill (safety for prod clusters)         |
| `noExitOnCtrlC`          | false   | require `:q` to quit (prevents fat-finger exits)                 |
| `headless`               | false   | hide top header                                                  |
| `crumbsless`             | false   | hide breadcrumb trail                                            |
| `logoless`               | false   | hide ASCII-art logo                                              |
| `enableMouse`            | false   | mouse clicks + scroll                                            |
| `reactive`               | false   | auto-reload when config files change                             |
| `logger.tail`            | 100     | lines on initial `l` view                                        |
| `logger.buffer`          | 1000    | max in-memory log lines before ring-buffer truncation            |
| `logger.sinceSeconds`    | -1      | default `--since=` for logs (-1 = tail forever)                  |

Example `~/.config/k9s/config.yaml` for a shared dev cluster:

```yaml
k9s:
  refreshRate: 5
  maxConnRetry: 5
  noExitOnCtrlC: true
  logger:
    tail: 400
    buffer: 5000
```

---

## 3. CLI flags (only the useful ones)

From the official commands page:

| flag                 | what it does                                                         |
|----------------------|----------------------------------------------------------------------|
| `-n <ns>`            | start in a specific namespace                                        |
| `-c <resource>`      | open directly into this resource view (e.g. `k9s -c rayjob`)        |
| `--context <name>`   | use a non-default kubecontext                                        |
| `--readonly`         | force read-only (ignore `config.yaml` toggle, good for prod)         |
| `help`               | print flags + exit                                                   |
| `info`               | print k9s runtime paths (logs, config dir, data dir) + exit          |

`k9s info` is especially handy when you're not sure which config.yaml k9s
is actually reading.

---

## 4. Command mode (`:`) — full list

Typing `:` at any view enters command mode. All of these work:

| syntax                               | effect                                                                     |
|--------------------------------------|----------------------------------------------------------------------------|
| `:<resource>`                        | switch to that resource view. Use plural, singular, or short name.         |
| `:<resource> <ns>`                   | same, filtered to a namespace (`:pod kube-system`)                         |
| `:<resource> /<regex>`               | open view pre-filtered by name regex (`:pod /ray-head`)                   |
| `:<resource> <label-selectors>`      | open view with label filter (`:pod training-type=40bra-md`)               |
| `:<resource> @<ctx>`                 | switch context *and* open this view                                        |
| `:ctx`                               | list + switch contexts                                                     |
| `:ctx <name>`                        | switch directly                                                            |
| `:ns`                                | list + switch namespaces                                                   |
| `:xray <resource> [ns]`              | XRay view — hierarchical tree of resource + what owns/uses it             |
| `:pulses` / `:pu`                    | cluster health pulse view                                                  |
| `:screendump` / `:sd`                | list previously saved yaml dumps (created with `ctrl-s`)                   |
| `:q`                                 | quit                                                                       |

k9s accepts any resource name the apiserver advertises, including CRDs.
That's why `:rayjob`, `:podgroup`, `:raycluster` all work without config.

Shortform aliases ship by default for the common core types — `:po` =
pods, `:dp` = deployments, `:sts` = statefulsets, `:no` = nodes, `:svc`
= services, etc. See `ctrl-a` at any moment for the live alias list.

---

## 5. Filter mode (`/`)

Inside *any* table view, `/` begins a filter. The filter is live — each
keystroke re-renders.

| syntax                 | what it matches                                                 |
|------------------------|-----------------------------------------------------------------|
| `/foo`                 | rows whose name matches regex `foo`                             |
| `/-l <label=value>`    | rows whose labels match (server-side label selector)            |
| `/-f <term>`           | fuzzy search over all visible columns                           |
| `/!foo`                | **inverse** — rows NOT matching `foo` (great for hiding noise)  |

Esc clears the filter. Filter text persists per-view until you clear.

---

## 6. Aliases (custom `:` commands)

Edit `~/.config/k9s/aliases.yaml`. Schema:

```yaml
aliases:
  <your-alias>: <group/version/resource>
```

Concrete example for this repo:

```yaml
aliases:
  # our custom resources
  rj: ray.io/v1/rayjobs
  rc: ray.io/v1/rayclusters
  pg: scheduling.volcano.sh/v1beta1/podgroups

  # pre-filtered views (just like :pod … — append ns/labels)
  mymd: ray.io/v1/rayjobs default training-type=40bra-md
  mysd: ray.io/v1/rayjobs default training-type=40bra-sd
```

After editing, the file reloads automatically. `:mymd` now opens a
RayJob view pre-filtered to your md training runs.

---

## 7. Hotkeys (custom keybindings)

Edit `~/.local/share/k9s/hotkeys.yaml`. Schema:

```yaml
hotKeys:
  <key-name>:
    shortCut: <Key-combo>      # e.g. Shift-0, Alt-R
    description: <shown in ?>
    command: <command mode string>   # same syntax as the : prompt
```

Example:

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
    description: PodGroups (scheduler gang)
    command: podgroups
```

Avoid keys the built-in UI already uses (letters a–z, digits, arrows,
/, :, esc, ctrl-*). `Shift-<digit>` and `Alt-<letter>` are usually safe.

Press `?` at any time to see the current effective keybindings including
your custom ones.

---

## 8. Plugins (wrap external commands on a hotkey)

`~/.local/share/k9s/plugins.yaml` extends the action menu on a selected
row. Each plugin binds a key combo to a shell command, with templated
placeholders for the selected resource.

```yaml
plugins:
  tail-training-log:
    shortCut: Shift-T
    description: tail run.log inside head pod
    scopes:
      - pods
    command: sh
    background: false
    args:
      - -c
      - 'kubectl exec -n $NAMESPACE $NAME -- tail -f /root/myCodeLab/host/verl/ckpts/**/run.log'
```

Available template vars: `$NAME`, `$NAMESPACE`, `$CONTAINER`, `$RESOURCE_NAME`,
`$RESOURCE_GROUP`, `$CLUSTER`, `$CONTEXT`, `$KUBECONFIG`, `$COL-<index>`
(the value of the N-th visible column).

`scopes` restricts the plugin to views where it makes sense. Use
`all` to offer it everywhere.

---

## 9. Hotkeys that come built-in (daily set)

These are the ones I reach for dozens of times a day:

### Navigation
- `:` — command mode
- `/` — filter
- `<esc>` — leave current mode / go back
- `?` — help for current view (shows ALL key bindings including plugins + hotkeys.yaml additions)
- `ctrl-a` — list all known resource aliases
- `ctrl-s` — screen-dump selected resource to yaml on disk

### Row actions
- `enter` — drill into children (rayjob → pods)
- `d` — describe
- `y` — view full yaml
- `e` — edit (opens `$EDITOR` / `$KUBE_EDITOR`)
- `l` — logs
- `p` — previous (crash-loop) logs
- `f` — port-forward (in pod / service view)
- `s` — shell into pod (kubectl exec -it)
- `ctrl-d` — delete (prompts)
- `ctrl-k` — kill (no prompt — dangerous)
- `m` — mark row (multi-select)
- `space` — select row (multi-select)

### Scroll / pagination in tables
- `j`/`k` or `↓`/`↑` — row
- `ctrl-d` in logs-view = page down (overloaded; context-sensitive)
- `g`/`G` — top / bottom
- `0` — reset to default namespace filter
- `1` … `9` — jump to the Nth namespace in the quick bar

### Log-view extras (after pressing `l`)
- `f` — follow/unfollow
- `w` — toggle wrap
- `t` — toggle timestamps
- `s` — save to disk (uses screen-dumps dir)
- `/` — filter log lines
- `0`/`1`/`2`/`3` — tail last 1 / 5 / 10 / 25 min
- `4`/`5` — last 1 h / all since start

---

## 10. Two built-in views most people overlook

### Pulses (`:pu`)

Cluster-wide health at a glance: top-line counts of pods, events, CPU
and memory pressure. If pods start failing you'll see the red bar move
before the training log tells you.

### XRay (`:xray <resource>`)

A dependency tree, not a table. `:xray rayjob` shows each RayJob
expanded with its RayCluster → Pods → PodGroup as nested children. Best
view to understand why a RayJob's deletion didn't cascade.

---

## 11. Read-only mode (safety rail)

On a shared / prod cluster, start with:

```bash
k9s --readonly
```

Or put in config.yaml:

```yaml
k9s:
  readOnly: true
```

All destructive hotkeys (delete, kill, edit) are disabled. You can still
describe, log, exec, port-forward, screen-dump.

---

## 12. Troubleshooting

| symptom                                                | check                                                                              |
|--------------------------------------------------------|------------------------------------------------------------------------------------|
| `unable to connect` at startup                         | `kubectl cluster-info` works? `$KUBECONFIG` set? proxy envs blocking the API IP?  |
| colors look broken                                     | `export TERM=xterm-256color` before `k9s`                                          |
| editing opens wrong editor                             | set `$EDITOR` or `$KUBE_EDITOR`                                                    |
| rayjobs / CRDs show nothing                             | `kubectl api-resources \| grep rayjob` — apiserver must advertise the CRD         |
| k9s laggy, API quota hit                               | bump `refreshRate` to 5 or 10                                                      |
| custom hotkeys don't work                               | `~/.local/share/k9s/hotkeys.yaml` (not `~/.config/...`); press `?` to verify      |
| can't find config file                                 | `k9s info` prints all paths it uses                                                |

---

## 13. When to reach for kubectl instead

k9s is fantastic for browsing, ad-hoc triage, log-tailing, and
one-off edits. It's the wrong tool for:

- **Scripting / automation** — use kubectl + jq. Everything in this
  repo's `bspo_scripts/submit_*.sh` is kubectl-based by design.
- **Mass operations** on 100+ resources — k9s's multi-select is capped.
  Pipe `kubectl get -o json | jq | xargs kubectl` instead.
- **Deep yaml diffs** — `kubectl neat` + diff tools are better.
- **Headless CI / cron jobs** — k9s is a TUI; it won't run under CI.

Rule of thumb: **interactive = k9s; scripted = kubectl**.

---

## 14. Going deeper

Official docs, in recommended reading order:

1. `/topics/commands/` — all CLI flags and `:` commands
2. `/topics/config/` — config.yaml reference
3. `/topics/aliases/` — custom `:` aliases
4. `/topics/hotkeys/` — custom keybindings
5. `/topics/plugins/` — wrap shell commands as hotkeys
6. `/topics/columns/` — customise column layouts per resource
7. `/topics/skins/` — color themes
8. `/topics/rbac/` — minimum RBAC permissions k9s needs
9. `/topics/bench/` — benchmark a service (built-in hey wrapper)
10. `/topics/video/` — 3-minute demo video from the author

All linked from <https://k9scli.io>.
