# Context #

now let's get a new git branch and image file branch:

previously I mannually split Gym into MJ_NEMO_GYM, by removing
all Gym's server related code, and make MJ_NEMO_GYM a standalone
function set. the purpose is I don't have to install Gym, which
have many dependences, and requires to run standalone server for
each domain / source_data; I was dev my verl infra, I need
simplictic settings

but now as my dev finished, now I need to integrates Gym into my
verl image

pls check this repo rigorously for how to integrates nemo_gym into
verl: verl/examples/tutorial/nemo_gym

and how did I integrates my customized standalone functions
package MJ_NEMO_GYM the callstack:

st_verl_dockerfile/verl/plans/fapo/run_fapo_combo.sh
verl/verl/utils/dataset/rl_dataset.py
verl/verl/experimental/reward_loop/reward_manager/dapo.py

``` python
from mjnemogym import verl_compute_score
@register("dapo")
class DAPORewardManager(RewardManagerBase):
    """DAPO Reward Manager."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = verl_compute_score
```

u can find more detailed dev notes in: verl/plans/preproc_nemogym

# Context: data setup #

for test data, u can use hf dataset:
nvidia/Nemotron-3-Nano-RL-Training-Blend

but notice its not in verl's rl_dataset required format,
verl/verl/utils/dataset/rl_dataset.py
requires:
prompt (to be parsed by apply chat template)
...
extra_info

etc.

u can read verl/plans/preproc_nemogym/VERL_INTEGRATION.md
and preprocess_nemogym_blend_v5.py
as reference

but files above are for my customized package mjnemogym, for the
actual NemoGym package, u need to rigorously read @Gym and 
@verl/verl/utils/dataset/rl_dataset.py
@verl/verl/experimental/reward_loop/reward_manager/dapo.py

to come up with NemoGym version of integration plan md file


# Task #

Now u have 2 tasks, but u need to refine them simultaneously,
because they are dependent and u need severl iterates i guess:

1. with surgical edits to
@verl/verl/utils/dataset/rl_dataset.py
@verl/verl/experimental/reward_loop/reward_manager/dapo.py
to support multi-domain reward fn using NemoGym. Let's migrate
from customized packages to official NemoGym pacakge.

the final purpose is for any domains supported in MJ_NEMO_GYM, we
need to also be able to eval with official NemoGym package..

in order to finish this, u need to do 4 stages:

A. write a plan md file

B. download and preproc sample data, u don't have to preproc all

nvidia/Nemotron-3-Nano-RL-Training-Blend

blend data. but u must use the blend data as test case

C. surgical edits to
@verl/verl/utils/dataset/rl_dataset.py
@verl/verl/experimental/reward_loop/reward_manager/dapo.py

and may other related files to migrate our verl infra from
MJ_NEMO_GYM to official @Gym

D. maintain a dev mannual, there guaranteed to be many trial and
error iterations and bug fixes. I need u to detailed document all
things we do for this migragation

2. update the Dockerfile

the first reason I chose to mannual implement the MJ_NEMO_GYM
package is to avoid installing @Gym packages, since there might
be many potential conflicts

what I need is u do this in an interative way:

A. install @Gym in our dockerfile, and compare it with our
running verl image, I need an exhaustive list of packages diff,
making sure no deps in @dockerfile_update_plan.md

### SA-4: "DO NOT TOUCH" Protected Package List

and also minimal changes even to un protected package.

B. if anything conflicts happened, u should directly edit deps in
@Gym and made our own git branch and commit in that repo

so we install Gym with customized deps with full respect to our
verl image env

C. after iterations, u build the final production ready image,
and u generate a report, diff deps between Gym image and
MJ_NEMO_GYM image, so I know exact diffs between those images' deps


3. dev requirements:

document every step, every changes in a dev mannual md file, name
a new one

and write plan prior commencing 2 tasks, but keep iteration
methodology in mind and update them whenever new findings
invalidated our old plans and keep update dev mannuals
