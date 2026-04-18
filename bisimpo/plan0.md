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

Don't do until I answered ur questions
<!-- 3. an implementation plan: files / modules need to be changed to -->
<!--    implement bspo. the reason is there are more to change than -->
<!--    fiberpo, seems bspo use reward to re-weight loss, while -->
<!--    current core_algos.py lack those data passed in. for this, I -->
<!--    made a draft try, u can see -->

<!--    stash@{0} group loss in @verl 's git repo -->

<!--    but it's only primary and not well implemented and over -->
<!--    complicated for bspo, we can borrow experience from it, but -->
<!--    don't add the complexity. let's learn and write a plan for a -->
<!--    well-designed mandatory changes only implementation -->


