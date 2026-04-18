Now we continue with dev new algo

# Context #

1. in @verl , at fiberpo_ablation_single branch, we implemented our
customized algo called fiberpo, u may diff it with our nemo head
to first write a holistic analysis to see what's changed to
enable fiberpo on verl side, only author spacegoing and lichang93
is changed by me.

2. In current dir:

```
fiberpo_in_core_algos.py: torch version of fiberpo in verl's core_algos.py
fiberpo_loss_numpy.py: numpy version of fiberpo

bisim_eqs.tex: latex equations for the new bspo loss
```

3.0: important clarification:

(1). sorry for the confusion, but !!! closely pay attention here: the
${}^{(\text{GSPO})}\tilde\delta_\tau$ has nothing to do with gspo
loss, I only make it significant visually to remind us, hparams
here should be relatively comparable to gspo, but the loss has
nothing to do with gspo

(2). we are only implementing single domain for now. so if some
term is constant for single domain case, drop them. and for the
sake of debugging / simplicity, when never possible, use constant
over code vars, do not consider compatability to multi-domain for
now. because we are still at dev stage, my theory is not formal
yet, and need to keep implementation simple enough to debug /
ablate etc.

(3) 128 and 16 are constants for our mini_bsz and n_responses
actually used in verl training script, I only wirte them explicit
to let u understand them exactly, but they should be more
formally writting as g or domain etc.

3. my idea switching from fiberpo to bisim po, which uses
bi-simulation / wasserstein distance of pi new and pi old to
derive a new bspo loss

it's doc in bisim_eqs.tex. with 4 variants of implementations to
run to see which implementation wins

the 4 variants are documented other "objectives"

4. the relation between bisim_eqs and fiberpo code: I tried to
   keep notations aligned with fiberpo implementation, so u are
   familiar which notation should be calc with what data / verl's
   var name.

# Task #

1. pls write an annotated latex file first, this file should
   contain a. exact equations as bisim_eqs.tex, but in an order u
   like. current order is a bit messy I think. b. pseudo code for
   each variable's implementation, numpy style but with verl's
   actual name

2. write me a question list for me to answer in case
   bisim_eqs.tex + fiberpo code still not answer all info u need.

3. an implementation plan: files / modules need to be changed to
   implement bspo. the reason is there are more to change than
   fiberpo, seems bspo use reward to re-weight loss, while
   current core_algos.py lack those data passed in. for this, I
   made a draft try, u can see

   stash@{0} group loss in @verl 's git repo

   but it's only primary and not well implemented and over
   complicated for bspo, we can borrow experience from it, but
   don't add the complexity. let's learn and write a plan for a
   well-designed mandatory changes only implementation

   I think mentally the implementation can be splitted to 2
   parts:

   A. build the callstack for core_algos.py receive all reward
   related new vars. we need to:
     i. based on lessons learned, first design a minimal set of
   vars need to be passed to `bspo_loss` so it can be
   implemented
     ii. design the callstack, exhaustively list what inputs to
   get from for these vars, and which modules are affected to
   pierce through to pass these algos

   B. implementation of bspo. the loss should be easy to
   implement, the hard part is design of hparams and flags of
   variants of implementations (4 variants we mentioned above)

   for hparams: they should be compatible of hydra & yaml system
   we use now, taking special care of bash + hydra + yaml hell
   for special characters (capital letters, digit v.s. char, - _
   etc...), we prefer use yaml over bash, more robust in our
   current workflow.

   there are some hparams should equal to gspo, such as epsilon
   of clipping should be at gspo's scale, not grpo's 0.28 / 0.2
   scale. pls be aware of this for diff variants

   for flags of variants: we'll run ablation runs later, I prefer
   switching variant explicitly with flags, should be compatible
   with our kuberay workflow we implemented above.

4. go implement and follow dev guide in
   iter_kuberay_32nodes_verl_training/ and 0418/. there should be
   2 stages: dev&debug and ablation run.

   dev & debug: I think even 5 grad steps is more than we need,
   maybe 1 step is enough for dev purpose?

   ablation run: u need to first schedule all combos to run in
   our combo yaml, then queue all jobs for them to run and
   complete. when design combos, u need to pay special attention
   to compare apple to apple.


# Notes #

1. keep everything / stage / changes / fixes / bugs / decisions
   documented in a dev_notes_[stage].md
2. I'm going to sleep, keep research and implement and debug
   until final run get obj comparable results, do not ask me
   anything from now on. do till u can decide which obj wins with
   sufficient hparams sweep and stats based evidence, don't
   forget to change eval step to 10 steps, and use 120 max steps
   for now. and u can read logs/ wandb offline data to get run
   stats. altough u might need to try to figure out how to parse
   wandb offline data correctly, a sample is here:
   @../bisimpo/wandb_parse_offline.py
