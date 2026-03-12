# Dev Manual: Gym Server-Mode Migration (iter2)

**Started**: 2026-03-12
**Goal**: Migrate from iter1 (offline function calls) to iter2 (Gym as local HTTP servers)
**Base Image**: `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev`

---

## Context

iter1 imported Gym's scoring functions directly as Python modules (`from resources_servers.code_gen...`).
iter2 runs Gym resource servers as local FastAPI services with `/verify` endpoints.

**Why server mode**:
- Extensibility: add new domains by deploying new servers, not rewriting adapter code
- Standard interface: all domains use the same `/verify` HTTP contract
- Scalability: each pod runs its own local servers (future 64-node jobs)
- Official pattern: follows `verl/examples/tutorial/nemo_gym/` reference

---

## Requirements Analysis

### verl Side (Reward Manager)

The reference pattern is `GoogleSearchRewardManager` in `verl/examples/tutorial/nemo_gym/reward_manager.py`.

**Key observations**:
1. Extends `AbstractRewardManager` with `@register("name")`
2. `__init__` receives `resource_server_url` from config
3. `__call__` iterates over DataProto items, decodes response text, calls `await self.compute_score()`
4. `compute_score` POSTs to `{resource_server_url}/verify` with JSON body
5. The JSON body includes minimal stubs for `responses_create_params` and `response`, plus domain-specific fields
6. Returns `resp_json["reward"]`

**Verify request JSON format** (from reference):
```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": ""}]
  },
  "response": {
    "id": "", "created_at": 0, "model": "", "object": "response",
    "parallel_tool_calls": false, "tool_choice": "auto", "tools": [],
    "output": [{"role": "assistant", "content": "<model_output>"}]
  },
  "<domain_specific_fields>": "..."
}
```

**Multi-domain difference from reference**: GoogleSearch uses a single server URL. Our blend has 5 domains, each with its own server. The reward manager needs a `{data_source: server_url}` routing map.

### Gym Side (Resource Servers)

Each domain's `/verify` endpoint expects a Pydantic model extending `BaseVerifyRequest`:

| Domain | Verify Request Fields (beyond base) | Returns |
|--------|-------------------------------------|---------|
| code_gen | `verifier_metadata: {unit_tests: {inputs, outputs, fn_name}}` | `reward: 0.0/1.0`, `extracted_model_code`, `result` |
| mcqa | `expected_answer`, `options`, `grading_mode`, `template_metadata`, `uuid` | `reward: 0.0/1.0`, `extracted_answer` |
| instruction_following | `id`, `instruction_id_list`, `prompt`, `kwargs`, `grading_mode` | `reward: 0.0/1.0`, `follow_instruction_list` |
| structured_outputs | `schema_str`, `schema_type` | `reward: 0.0/1.0` |
| workplace_assistant | `ground_truth`, `id`, `category`, `environment_name` | `reward: 0.0/1.0` |

**NeMoGymResponse format**: The servers extract model output text from `response.output[-1].content[0].text` (for message items) or from function_call items (for workplace). The `responses_create_params` field is required by Pydantic but only echoed — not used in scoring logic.

### Raw Data Schema (train.jsonl)

Each row has different keys per domain. Common field: `agent_ref.name` → determines domain.

| Agent Name | Domain Keys | Missing in current extra_info |
|------------|-------------|-------------------------------|
| `code_gen_simple_agent` | `verifier_metadata` (with `unit_tests`) | Already in extra_info |
| `mcqa_simple_agent` | `expected_answer`, `options`, `template_metadata`, `grading_mode` | Already in extra_info |
| `instruction_following_simple_agent` | `id`, `instruction_id_list`, `prompt`, `kwargs` | Already in extra_info |
| `structured_outputs_simple_agent` | `schema_str`, `schema_type` | Already in extra_info |
| `workplace_assistant_simple_agent` | `ground_truth`, `id`, `category`, `environment_name` | Already in extra_info |

**Key finding**: The current preprocessor (`preprocess_nemogym_blend_v5.py`) already puts ALL domain-specific fields into `extra_info`. No preprocessing changes needed — the existing parquet works for iter2.

### Processed Parquet Schema (val.parquet)

5 columns: `data_source`, `prompt`, `reward_model`, `ability`, `extra_info` (JSON string)

Distributions (val): nemogym_mcqa=27, nemogym_code=26, nemogym_if=26, nemogym_workplace=14, nemogym_structured=7

---

## Implementation Plan

### Task 1: Gym Server Launcher Script

Create `my_scripts/launch_gym_servers.sh` to start all 5 resource servers locally.

Each server runs via uvicorn on a unique port:
```
code_gen:               port 19001
mcqa:                   port 19002
instruction_following:  port 19003
structured_outputs:     port 19004
workplace_assistant:    port 19005
```

Approach: Start each server directly with `python -m uvicorn` using the Gym server's FastAPI app.
Alternative: Use Gym's `SimpleServer.run_webserver()` classmethod which handles config + uvicorn internally.

**Decision**: Use standalone uvicorn launch scripts since we need `--no-deps` Gym and want fine-grained control. Each server gets a small Python wrapper that creates the server instance and runs uvicorn.

### Task 2: Multi-Domain Reward Manager

Create `verl/verl/workers/reward_manager/nemogym_server.py` (new file) following GoogleSearchRewardManager pattern.

**Design**:
```python
@register("nemogym_server")
class NemoGymServerRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key,
                 server_urls: dict[str, str]):
        # server_urls = {
        #   "nemogym_code": "http://localhost:19001",
        #   "nemogym_mcqa": "http://localhost:19002",
        #   ...
        # }

    async def compute_score(self, data_source, solution_str, ground_truth, extra_info):
        server_url = self.server_urls[data_source]
        verify_body = self._build_verify_request(data_source, solution_str, extra_info)
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{server_url}/verify", json=verify_body) as resp:
                return (await resp.json())["reward"]

    def _build_verify_request(self, data_source, solution_str, extra_info):
        # Minimal stubs for required Pydantic fields
        base = {
            "responses_create_params": {"input": [{"role": "user", "content": ""}]},
            "response": {
                "id": "", "created_at": 0, "model": "", "object": "response",
                "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
                "output": [{"role": "assistant", "content": solution_str}],
            },
        }
        # Add domain-specific fields from extra_info
        if data_source == "nemogym_code":
            base["verifier_metadata"] = extra_info.get("verifier_metadata")
        elif data_source == "nemogym_mcqa":
            base["expected_answer"] = extra_info.get("expected_answer")
            base["options"] = extra_info.get("options")
            base["grading_mode"] = extra_info.get("grading_mode", "strict_single_letter_boxed")
            if "template_metadata" in extra_info:
                base["template_metadata"] = extra_info["template_metadata"]
        # ... etc for each domain
        return base
```

**Fallback**: Non-nemogym data_sources fall through to `default_compute_score` (same as iter1).

### Task 3: Update Run Script

Modify `my_scripts/run_moonlight_1node_blend_smoke.sh`:

1. Add server URLs as env vars or CLI args
2. Change `reward_model.reward_manager=dapo` → `reward_model.reward_manager=nemogym_server`
3. Pass `+reward_model.reward_kwargs.server_urls=...` (dict of data_source→URL)
4. Add `custom_reward_function.path` and `custom_reward_function.name` if needed
5. Add server startup (call `launch_gym_servers.sh`) before training

### Task 4: Dockerfile Changes

**New deps needed for Gym server mode** (not just scoring functions):
- `fastapi` — server framework
- `uvicorn` — ASGI server
- `aiohttp` — async HTTP client (for reward manager)
- `uvloop` — optional but recommended for perf

**Changes from iter1**:
- iter1: `pip install --no-deps -e Gym` (only scoring functions)
- iter2: `pip install -e Gym` with server deps OR `pip install --no-deps -e Gym` + explicit server deps

**Decision**: Keep `--no-deps` for Gym but add explicit server deps to avoid pulling unwanted packages (mlflow, openai client, etc.).

Add to pip layer:
```
fastapi uvicorn[standard] aiohttp
```

Check against DO NOT TOUCH list: none of these conflict with torch/vllm/flash_attn. **Safe.**

### Task 5: gym_compute_score.py — Keep as Fallback

Keep the existing `gym_compute_score.py` for the legacy (non-RewardLoop) code path and as fallback for `dapo` reward manager with non-nemogym data sources. The new `nemogym_server.py` reward manager is the primary path for blend training.

### Task 6: Server Wrapper Scripts

For each domain, create a minimal launch script:

```python
# launch_code_gen_server.py
import uvicorn
from resources_servers.code_gen.app import CompCodingResourcesServer

config = CompCodingResourcesServerConfig(
    host="0.0.0.0", port=19001,
    num_processes=4, unit_test_timeout_secs=10, debug=False,
)
server = CompCodingResourcesServer(config=config)
app = server.setup_webserver()
uvicorn.run(app, host="0.0.0.0", port=19001)
```

Or simpler: use `SimpleServer.run_webserver()` with env-based config.

**Open question**: Whether Gym servers need Ray. `code_gen` uses `ray.remote` for subprocess execution. If so, they can share verl's Ray cluster or we launch a separate local-only Ray.

---

## Data Pipeline (No Changes)

Existing preprocessor and parquet format work for iter2:
- `extra_info` already contains all domain-specific fields needed for `/verify`
- `data_source` maps to the correct server
- `prompt` is already normalized for verl's chat format

---

## Dependency Diff (iter1 → iter2)

| Package | iter1 | iter2 | Reason |
|---------|-------|-------|--------|
| Gym (editable) | `--no-deps` | `--no-deps` | Same |
| fastapi | In base | In base | Already available |
| uvicorn | In base | In base | Already available |
| aiohttp | In base | In base | Already available |
| yappi | Not needed | Added (L5) | Gym profiling module |
| itsdangerous | Not needed | Added (L5) | Starlette SessionMiddleware |
| gprof2dot | Not needed | Added (L5) | Gym profiling module |
| pydot | Not needed | Added (L5) | Gym profiling dep |
| verifiable-instructions | Needed | Needed | IF scoring (used by IF server) |
| math_verify | Needed | Needed | Math scoring (used by math server) |
| openapi_schema_validator | Needed | Needed | SO scoring |
| langdetect, absl-py, immutabledict | Needed | Needed | verifiable-instructions deps |

---

## Risk Assessment

1. **Gym server startup time**: FastAPI servers take a few seconds to start. Script must wait for health before training.
2. **code_gen server needs Ray**: Uses `ray.remote` for subprocess isolation. May conflict with verl's Ray cluster.
3. **NeMoGymResponse Pydantic validation**: The stub format must pass Pydantic's strict validation. Need to test.
4. **Concurrency**: Multiple training workers hitting verify endpoints concurrently. FastAPI handles this but code_gen uses semaphore.
5. **Port conflicts**: 5 servers on ports 19001-19005. Must not conflict with Ray (6379), vLLM, etc.

---

## Changelog

| Step | Status |
|------|--------|
| Analyze raw JSONL schema | Done |
| Analyze processed parquet schema | Done |
| Analyze Gym verify endpoint schemas | Done |
| Analyze verl reward manager reference pattern | Done |
| Write iter2 migration plan | Done |
| Copy datasets to local dir (not symlinks) | Done |
| Fix code_gen bare lcb_integration imports | Done |
| Create gym_server_runner.py (standalone server launcher) | Done |
| Create launch_gym_servers.sh (5 servers on ports 19001-19005) | Done |
| Create nemogym_server.py reward manager | Done |
| Register nemogym_server in __init__.py | Done |
| Update Dockerfile: add yappi, itsdangerous, gprof2dot, pydot | Done |
| Update run script: nemogym_server + server_urls | Done |
| Test: all 5 server instantiation | Done |
| Test: verify endpoint per domain (mcqa, if, so, wa, code) | Done (all return correct rewards) |
| Build image | Pending |
| Test: end-to-end blend smoke | Pending |
