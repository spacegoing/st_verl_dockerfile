I've read iter1's dev_manual_gym_migration.md

u completely miss understood my requirements

by mitigrating to gym, I mean using gym as server mode, but that
server run in the same image locally. so say in future I have 64
nodes job, on that 64 pods each requires their own local
container / pod for evaluation

actually pls track the call stack of verl reward call stack, I
suddenly realized, the reward seems to be graded by a single
controller? so all dp ranks send response back to rank0 node for
rewards?

either way, figure this out first, write a callstack reward
analysis md file. let's not tackle this since we are only run on
one node now

but I need u to use gym as it should be, the reason we mitigrate
to gym is we are about to largely extend our domains, so we can't
use adhoc code anymore, has to use nemo gym as it is

it is totally wanted writing a verl class such as
gym_compute_score.py to wrap interface to gym, keep doing this.

but u need to refer to @verl/examples/tutorial/nemo_gym for how
to integrates official gym to verl and make our design really
easy to extend to arbitrarily many other domains
(6 domains in blend dataset already)

update ur plan to iter2 mitigration plan md file, and then
reimplement.
