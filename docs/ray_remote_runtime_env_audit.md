# Ray `@ray.remote` `runtime_env` Audit

All `@ray.remote` usages in the Gym repo, with analysis of whether each needs
`py_executable` and/or `PYTHONPATH` in `runtime_env`.

**Background**: Two distinct problems, two distinct fixes:
- `py_executable: sys.executable` — workers use the **server's venv** Python, not system Python
- `env_vars: {PYTHONPATH: <dir>}` — workers can find packages that are **not pip-installed** (plain directories)

Workers spawn in a temp directory and inherit neither the server's cwd nor its `sys.path`.

---

## Summary Table

| File | Type | `py_executable` | `PYTHONPATH` | Status |
|------|------|----------------|--------------|--------|
| `responses_api_agents/harbor_agent/app.py` | task | ✅ | not needed | ✅ OK |
| `responses_api_agents/mini_swe_agent/app.py` | task | ✅ | not needed | ✅ OK |
| `responses_api_agents/swe_agents/app.py` | task | ✅ | not needed | ✅ OK |
| `responses_api_models/local_vllm_model/app.py` | Actor | intentionally absent | not needed | ✅ OK (see note) |
| `resources_servers/code_gen/lcb_integration/compute_code_generation_metrics.py` | task | ✅ | ✅ | ✅ Fixed 2026-03-19 |
| `resources_servers/swerl_gen/eval/singularity_utils.py` | task | ✅ | ✅ | ✅ Fixed 2026-03-19 |

---

## Detailed Analysis

### 1. `responses_api_agents/harbor_agent/app.py:181`

```python
@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={"py_executable": sys.executable},
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)
```

**What it does**: Dispatches a `runner` callable (passed as argument, cloudpickle-serialized)
to a remote worker. The callable contains the actual harbor job logic.

**`py_executable`**: Needed ✅ — ensures the worker runs in harbor_agent's per-server venv,
not system Python.

**`PYTHONPATH`**: Not needed — all imports inside `runner` come from packages properly
pip-installed in the harbor_agent venv (`nemo_gym`, `fastapi`, `aiohttp`, etc.).
No plain-directory packages like `lcb_integration`.

---

### 2. `responses_api_agents/mini_swe_agent/app.py:78`

```python
@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={"py_executable": sys.executable},
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)
```

**What it does**: Identical pattern to harbor_agent — dispatches a runner callable.

**`py_executable`**: Needed ✅ — uses mini_swe_agent's per-server venv.

**`PYTHONPATH`**: Not needed — all deps are pip-installed in the venv.

---

### 3. `responses_api_agents/swe_agents/app.py:59`

```python
@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={"py_executable": sys.executable},
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    ray_submit_time = time.time()
    params["ray_submit_time"] = ray_submit_time
    return asyncio.run(runner(**params))
```

**What it does**: Same runner-dispatch pattern; async variant (`asyncio.run`).

The module imports `from responses_api_agents.swe_agents.utils import ...` at the top.
These are part of the installed Gym package (`nemo-gym[dev]` via `-e .`), so all submodules
under `responses_api_agents.*` are importable from the venv without PYTHONPATH.

**`py_executable`**: Needed ✅ — uses swe_agents' per-server venv.

**`PYTHONPATH`**: Not needed — Gym is installed as an editable package; all `responses_api_agents.*`
imports resolve through the normal package installation.

---

### 4. `responses_api_models/local_vllm_model/app.py:73`

```python
@ray.remote
class LocalVLLMModelActor:
    def __init__(self, head_node_placement_group, server_args, env_vars, ...):
        from vllm.entrypoints.openai.api_server import run_server
        ...
```

**What it does**: A Ray **Actor** (long-lived stateful process, not a one-shot task) that
runs a vLLM server in a background thread. It's a fundamental architectural difference from
the task functions above.

**`py_executable`**: Intentionally absent — `sys.executable` here would be
`local_vllm_model`'s per-server venv Python, which does **not** have vLLM installed
(vLLM is only in the system venv from the base image). Using `py_executable: sys.executable`
here would **break** the Actor. The correct behavior is for Ray to use system Python,
which has vLLM. This is safe in our setup because all cluster nodes run the same container
image with vLLM in system Python.

**`PYTHONPATH`**: Not needed — vLLM and all other imports are pip-installed in system Python.

**Note**: This is a deliberate design choice, not a missing fix. Adding `py_executable` would
cause `ImportError: No module named 'vllm'`.

---

### 5. `resources_servers/code_gen/lcb_integration/compute_code_generation_metrics.py:52`

```python
_CODE_GEN_DIR = str(Path(__file__).parent.parent)  # → .../code_gen/

@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
        "env_vars": {"PYTHONPATH": _CODE_GEN_DIR},
    },
)
def check_correctness_remote(sample, generation, timeout, debug=True):
    return check_correctness(sample, generation, timeout, debug)
```

**What it does**: Runs code correctness checks (Python execution + test case evaluation)
on remote workers, spread across nodes for parallelism.

**`py_executable`**: Needed ✅ — uses code_gen's per-server venv (has `lcb_integration`
runtime deps like `numpy`).

**`PYTHONPATH`**: Needed ✅ — `lcb_integration` is a plain directory with no
`pyproject.toml`; it cannot be pip-installed. Workers spawn in a temp dir and cannot
find it without an explicit path. `_CODE_GEN_DIR` is computed from `Path(__file__).parent.parent`
at module import time (when the server starts), so it always resolves to the live path
in the running container.

**Status**: Fixed 2026-03-19. Previously used a `ln -sf` symlink into
`/usr/local/lib/python3.12/dist-packages/` — wrong approach (used system Python,
not code_gen venv; also fragile at container startup).

---

### 6. `resources_servers/swerl_gen/eval/singularity_utils.py:203`

```python
@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
        "env_vars": {"PYTHONPATH": "/opt/nemo-rl/3rdparty/Gym-workspace/Gym"},
    },
)
def compute_score(extra_info_base64, patch_str, repro_test_info_base64, mode, ...):
    return calculate_execution_feedback_reward(...)
```

**What it does**: Runs SWE-bench evaluation inside a Singularity container via subprocess.

**`py_executable`**: Needed ✅ — uses swerl_gen's per-server venv.

**`PYTHONPATH`**: Needed ✅ — the worker's `compute_score` calls `calculate_execution_feedback_reward`
(same file), which imports `from resources_servers.swerl_gen.eval.reward_functions import ...`.
Even though Gym is installed as `-e .`, the per-server venv may not resolve `resources_servers.*`
without the Gym root on `PYTHONPATH`. The hardcoded path `/opt/nemo-rl/3rdparty/Gym-workspace/Gym`
points to the Gym root in swerl_gen's target deployment environment.

**Status**: `PYTHONPATH` was already present. Fixed 2026-03-19 by adding missing
`py_executable: sys.executable`.

---

## Docs/Template Files (not real code)

- `.claude/skills/add-benchmark/references/patterns.md:410` — example snippet in a skill template
- `docs/reference/faq.md:366` — documentation example

Neither is executed; no fix needed.

---

## `build/` Directory

`Gym/build/lib/resources_servers/swerl_gen/eval/singularity_utils.py` is a Python build
artifact (copy produced by `python setup.py build`). It is stale and not executed at runtime.
The fix should only be applied to the source file under `resources_servers/`, which was done.
