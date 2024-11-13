Project objective: Get a (robust as possible) subset of the most important wavebands for predicting soil gene abundance values.

Constraints:
1. Python instead of R
2. Only elastic net models (random forest had bad results previously, and elastic net is a generalization of lasso models so those are implicitly included in consideration)

Subgoals:
1. Finish implementing basic modeling in scikit-learn
	a. Figure out why the existing elastic net model isn't getting good results. Probably have to do hyperparameter training from scratch instead of transferring in prior HPs.
	b. Try to get as robust of a model as possible
2. Implement feature selection methods to give a set of the top x wavebands
	a. Filter methods: These are applied to the data based on its statistical properties. No modeling needed.
		i. Correlation filter threshold?
		ii. Chi-squared threshold? (This one suspect since it may be for categorical, not numerical data. Need to look into this)
	b. Embedded methods: These use internal properties of the models themselves
		i. Coefficients for elastic net
	c. Wrapper methods: These are model-agnostic, and (according to a few sources) generally the most robust out of the three types.
		i. Recursive feature elimination
		ii. Permutation importance
3. Get a way to algorithmically find a consensus among the waveband sets from part 2
4. Repeat part 1, but on the results of part 3
5. Analyze results
6. Write up paper (doubles work for MLSC and for SoutheastCon)
7. Create presentation