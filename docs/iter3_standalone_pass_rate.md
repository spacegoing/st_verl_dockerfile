pls read @Gym's docs thoroughly.

the purpose is that, now in @verl/plans/nemo_gym_worker we have
given experience how to build standalone reward server, and under
@readme.md we have built gym image with isolated venv

we also have training dataset in @verl/my_scripts/

as documented in gym, @Gym/docs/model-server/vllm.md and
@Gym/docs/get-started/rollout-collection.md it can collect rollouts

with even hermes parser and think parser enabled, which we
documented concerns in verl implementation

now what I want is a pass rate generator:

it will load all questions in the train_v2.parquet as used in
@verl/my_scripts/run_moonlight_1node_blend_smoke.sh

and how that is generated u can read @readme.md and @dev_notes.md

what I want is for each prompt in that parquet, u rollout 10
rollouts and evaluate them using the moonlight model in
@verl/my_scripts/run_moonlight_1node_blend_smoke.sh
and using nemo gym respective resource servers to calc accuracy.
all 6 domains are the same as used in verl

and u calc a moonlight_pass_acc@10 key in the parquet for each
question, which is the average reward / score for all 10 samples
scored by the resource server.

u write 2 parquet files:

1. the parquet with moonlight_pass_acc@10 appended, everything
   else same with train_v2.parquet
2. the parquet with all 10 samples for each prompt, together with
   returned scores
3. do not change train_v2.parquet

pls:

1. create a new gym_rollout/ sub dir for this project, and write
   everything there.
2. write a new dockercompose file for this task. do not use vrl
   container, these are 2 different tasks
3. as before, keep the whole dev process updated / documented in
   a dev note md file so we know everything along the way
4. pls debug till everything works
