**TODO: Modify this readme once the manuscript is done, and change it to something more descriptive**

Project objective: Get a (robust as possible) subset of the most important wavebands for predicting soil gene abundance values.

Constraints:
1. Python instead of R
2. Only elastic net models (random forest had bad results previously, and elastic net is a generalization of lasso models so those are implicitly included in consideration)

Subgoals:
1. Finish implementing basic modeling in scikit-learn
    1. Figure out why the existing elastic net model isn't getting good results. Probably have to do hyperparameter training from scratch instead of transferring in prior HPs.
    2. Try to get as robust of a model as possible
2. Implement feature selection methods to give a set of the top x wavebands
    1. Filter methods: These are applied to the data based on its statistical properties. No modeling needed.
        1. Correlation filter threshold?
        2. Chi-squared threshold? (This one suspect since it may be for categorical, not numerical data. Need to look into this)
    2. Embedded methods: These use internal properties of the models themselves
        1. Coefficients for elastic net
    3. Wrapper methods: These are model-agnostic, and (according to a few sources) generally the most robust out of the three types.
        1. Recursive feature elimination
        2. Permutation importance
3. Get a way to algorithmically find a consensus among the waveband sets from part 2
4. Repeat part 1, but on the results of part 3
5. Analyze results
6. Write up paper (doubles work for MLSC and for SoutheastCon)
7. Create presentation

Notes:
- <https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py> gives an example of hyperparameter searching over a pipeline, which is already what we need to do. But further, it even tests multiple dimensionality reduction methods simultaneously. Is this basically what we want? I still need to look into it.