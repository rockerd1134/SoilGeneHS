#!/usr/bin/env python
# coding: utf-8

# In[1]:


import git
import numpy as np
import pandas as pd
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

from sklearn_genetic import GAFeatureSelectionCV


# In[2]:


repo = git.Repo('.', search_parent_directories = True)
root = repo.working_tree_dir

SEED = 0
rng = np.random.default_rng(SEED)

# Intended for reproducible GA steps
np.random.seed(SEED)
random.seed(SEED)


# **Preprocessing**

# In[3]:


data_consol = pd.read_csv(root + '//data/data_consol.csv')

X = data_consol.filter(regex="^[0-9]+$")
    
# Note: do NOT scale X and y before splitting, since that is a data leak. Instead, use the pipeline to scale both Xs, and separately scale the y for custom scoring like RMSE.
X_all_train, X_all_test, y_train_unscaled, y_test_unscaled = train_test_split(data_consol.filter(regex="^[0-9]+$").to_numpy(), data_consol.filter(regex="pcr_[a-z]+_log"), train_size=0.8, random_state=0)

# Separate y by genes. Reshaping necessary for the y scaling step
bact_train_unscaled = y_train_unscaled["pcr_bact_log"].to_numpy().reshape(-1,1)
bact_test_unscaled = y_test_unscaled["pcr_bact_log"].to_numpy().reshape(-1,1)

cbblr_train_unscaled = y_train_unscaled["pcr_cbblr_log"].to_numpy().reshape(-1,1)
cbblr_test_unscaled = y_test_unscaled["pcr_cbblr_log"].to_numpy().reshape(-1,1)

fungi_train_unscaled = y_train_unscaled["pcr_fungi_log"].to_numpy().reshape(-1,1)
fungi_test_unscaled = y_test_unscaled["pcr_fungi_log"].to_numpy().reshape(-1,1)

urec_train_unscaled = y_train_unscaled["pcr_urec_log"].to_numpy().reshape(-1,1)
urec_test_unscaled = y_test_unscaled["pcr_urec_log"].to_numpy().reshape(-1,1)

# Special case: phoa has 10 NAN rows that need to be removed from both its X and y.
phoa_data = data_consol.filter(regex="^[0-9]+$|pcr_phoa_log").dropna()
X_all_phoa = phoa_data.to_numpy()[:,:2151]
phoa = phoa_data["pcr_phoa_log"].to_numpy()
X_all_phoa_train, X_all_phoa_test, phoa_train_unscaled, phoa_test_unscaled = train_test_split(X_all_phoa, phoa, train_size=0.8, random_state=0)
phoa_train_unscaled = phoa_train_unscaled.reshape(-1,1)
phoa_test_unscaled = phoa_test_unscaled.reshape(-1,1)

# Create copies of the X sets for visible light only (using bounds of 400 nm -> 700 nm)
X_vis_train = X_all_train[:,400:701]
X_vis_test = X_all_test[:,400:701]
X_vis_phoa_train = X_all_phoa_train[:,400:701]
X_vis_phoa_test = X_all_phoa_test[:,400:701]

# Scale each y with respect to its distribution
bact_scaler = StandardScaler()
bact_train = bact_scaler.fit_transform(bact_train_unscaled).reshape(-1,1)
bact_test = bact_scaler.transform(bact_test_unscaled).reshape(-1,1)

cbblr_scaler = StandardScaler()
cbblr_train = cbblr_scaler.fit_transform(cbblr_train_unscaled).reshape(-1,1)
cbblr_test = cbblr_scaler.transform(cbblr_test_unscaled).reshape(-1,1)

fungi_scaler = StandardScaler()
fungi_train = fungi_scaler.fit_transform(fungi_train_unscaled).reshape(-1,1)
fungi_test = fungi_scaler.transform(fungi_test_unscaled).reshape(-1,1)

phoa_scaler = StandardScaler()
phoa_train = phoa_scaler.fit_transform(phoa_train_unscaled).reshape(-1,1)
phoa_test = phoa_scaler.transform(phoa_test_unscaled).reshape(-1,1)

urec_scaler = StandardScaler()
urec_train = urec_scaler.fit_transform(urec_train_unscaled).reshape(-1,1)
urec_test = urec_scaler.transform(urec_test_unscaled).reshape(-1,1)

# 5-fold CV; random state 0
cv_5_0 = KFold(n_splits=5, shuffle=True, random_state=0)

# Used for waveband selection
wvs = np.arange(350,2501)
wvs_vis = np.arange(400,701)


# **The major pipeline components**

# In[4]:


elastic_net = ElasticNet(fit_intercept=False, warm_start=True, random_state=0, selection='random', max_iter=8000)

# Used for embedded feature importance (via coeffs) and wrapper feature importance (via perm importance)
pipe_elastic_net = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("elastic_net", elastic_net)
    ],
    verbose=False
)

# Hyperparameters for elastic net tuning. When code is finalized, expand for more thorough search using more computational resources.
REGULARIZATION = np.logspace(-5, 5, 16)
MIXTURE = np.linspace(0.001, 1, 16)
PARAM_GRID = [
    {
        "elastic_net__alpha": REGULARIZATION,
        "elastic_net__l1_ratio": MIXTURE
    }
]


# **The baseline models (for comparison)**

# In[5]:


def baseline(X_train, y_train, X_test):
    """ Build an elastic net model on the whole set of wavebands considered (no waveband selection methods used). 
    Returns the predictions on y_train."""
    print('\tStarting baseline...', end='')
    model = GridSearchCV(estimator=pipe_elastic_net, param_grid=PARAM_GRID, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv_5_0, error_score='raise')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('Done.')
    return preds, model


# **The feature selection functions**

# In[6]:


# Since this is only with respect to X_train, not any of the target variables, this only has to be computed once. (It's relatively cheap to compute, but this also has the benefit of preserving the random choices.)
def cluster(X_train, region):
    """ Uses agglomerative clustering with a distance threshold of 0.999 on the normalized feature correlation coefficient matrix. Then, it randomly selects one waveband from each cluster.
    This should be used as a preprocessing step when doing permutation importance. (Clustering method) """
    corr = np.corrcoef(X_train.T) # X needs to be transposed because of corrcoef's implementation
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0.999) # The distance threshold is somewhat arbitrary, but it's based on EDA and domain knowledge, and the results seem reasonable.
    clusters = agg.fit_predict(corr)
    # Now select a single "representative" waveband from each cluster
    if(region == "all"):
        wavebands = wvs
    elif(region == "vis"):
        wavebands = wvs_vis
    cluster_choices = []
    for i in range(np.max(clusters)):
        wv_in_cluster = wavebands[clusters==i]
        cluster_choices.append(rng.choice(wv_in_cluster))
    cluster_choices = np.sort(np.array(cluster_choices))
    return cluster_choices


# In[7]:


# Go ahead and call this once.
cluster_choices_all = cluster(X_all_train, "all")
cluster_choices_vis = cluster(X_vis_train, "vis")


# In[8]:


def mi(X_train, y_train, region, n_features=64):
    """ Uses mutual information to calculate the n_features most related features in X_train to y_train. (Filter method) """
    y_train = y_train.ravel()
    mi = mutual_info_regression(X_train, y_train)
    top_n_idx = np.argpartition(mi, -n_features)[-n_features:]
    if(region == "all"):
        return wvs[top_n_idx]
    elif(region == "vis"):
        return wvs_vis[top_n_idx]


# In[9]:


def train_elastic_net(X_train, y_train):
    """ Builds and fits an elastic net model using all features. 
    Returns the fit estimator (a pipeline). Used within coeffs() and ga(). """
    grid = GridSearchCV(estimator=pipe_elastic_net, param_grid=PARAM_GRID, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv_5_0, error_score='raise')
    grid.fit(X_train, y_train)
    return grid.best_estimator_


# In[10]:


def coeffs(estimator, region, n_features=64):
    """ Builds and fits an elastic net model using all features. Returns the n_features features with the highest absolute-valued coefficients. (Embedded method) """
    coeffs = estimator['elastic_net'].coef_
    abs_coeffs = np.abs(coeffs)
    top_n_idx = np.argpartition(abs_coeffs, -n_features)[-n_features:]
    if(region == "all"):
        return wvs[top_n_idx]
    elif(region == "vis"):
        return wvs_vis[top_n_idx]


# In[11]:


def ga(X_train, y_train, trained_estimator, wv_subset, n_features=64):
    """ Uses a genetic algorithm to find the wavebands that gives the lowest RMSE on an elastic net model. 
    The subset will be at most n_features large, but it may be less than n_features large. 
    wv_subset should be wvs (or wvs_vis) when in the feature selection layer, but when in the consolidation layer, it should
    be the subset of possible wavelengths output by the concatenated feature selection methods, not the entire
    [350,2500] (or [400,700]) set. (GA method) """
    
    y_train = y_train.ravel()
    ga_selector = GAFeatureSelectionCV(
        estimator=trained_estimator,
        cv=cv_5_0,  # Cross-validation folds
        scoring="neg_root_mean_squared_error",  # Fitness function (maximize accuracy)
        population_size=n_features*2,  # Number of individuals in the population
        generations=20,  # Number of generations
        n_jobs=-1,  # Use all available CPU cores
        verbose=False,
        max_features=n_features,
        return_train_score=True,
        refit=False,
        crossover_probability=0.8,
        mutation_probability=0.2
    )
    pipe_ga = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ga", ga_selector)
        ],
        verbose=False
    )
    
    pipe_ga.fit(X_train, y_train)
    feats = pipe_ga['ga'].best_features_ # A mask of the features selected from X_train
        
    return wv_subset[feats]


# In[12]:


def pi(X_train, y_train, region, n_features=64):
    """ Calculates permutation importance on a dataset. cluster_choices should be the result of calling cluster(), which should be done once at the start of execution. 
    This is done outside this function to preserve the random selection. Returns the set of n_features wavebands with the highest permutation importance on the training set. (Wrapper method) """
    # Use only the features selected by clustering
    if(region == "all"):
        cluster_choices = cluster_choices_all
        cluster_idx = cluster_choices - 350
    elif(region == "vis"):
        cluster_choices = cluster_choices_vis
        cluster_idx = cluster_choices - 400 # Needed since in vis, waveband 400 is the 0th.
    
    X_train = X_train[:,cluster_idx]
    # Build and train another elastic net model, but only on the features left after clustering, to use for permutation importance.
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("elastic_net", elastic_net)
        ], 
        verbose=False
    )    
    grid = GridSearchCV(estimator=pipe, param_grid=PARAM_GRID, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv_5_0, error_score='raise')
    grid.fit(X_train, y_train)
    pi = permutation_importance(grid, X_train, y_train, scoring='neg_root_mean_squared_error', n_repeats=10, n_jobs=-1, random_state=0)
    # Needed in case fewer than the threshold were chosen by the method
    n = min(pi.importances_mean.shape[0], n_features)
    pi_top_n_idx = np.argpartition(pi.importances_mean, -n)[-n:]
    return cluster_choices[pi_top_n_idx]


# **Consensus function**

# In[13]:


def consensus(X_train, y_train, region, n_features_intermed=64, max_features_output=16):
    """ Takes the wavebands output by the feature selection functions and uses a (separate) genetic algorithm to find the wavebands that give the lowest RMSE on an elastic net model.
    The subset will be at most n_features large, but it may be less than n_features large.
    Returns the tuple: (wv_mi, wv_coeffs, wv_ga, wv_cluster, wv_pi, wv_consensus), where each is a numpy array of wavebands that were selected by each method. """
    
    if(region == "all"):
        wv_subset = wvs # Used for GA
        wv_cluster = cluster_choices_all # Used when appending to wv_intermed
    elif(region == "vis"):
        wv_subset = wvs_vis
        wv_cluster = cluster_choices_vis
    
    print('\tStarting mutual importance...', end=' ')
    wv_mi = mi(X_train, y_train, region, n_features=n_features_intermed)
    print('Done.')
    print('\tTraining the elastic net model...', end=' ')
    trained_pipe = train_elastic_net(X_train, y_train)
    wv_coeffs = coeffs(trained_pipe, region, n_features=n_features_intermed)
    print('Done.')
    print('\tStarting genetic algorithm...', end=' ')
    wv_ga = ga(X_train, y_train, trained_pipe, wv_subset, n_features=n_features_intermed)
    print('Done.')
    print('\tStarting permutation importance...', end=' ')
    wv_pi = pi(X_train, y_train, region, n_features=n_features_intermed)
    print('Done.')

    # Compile the above results into one array, remove any duplicates, and sort.
    wv_intermed = np.append(wv_mi, wv_coeffs)
    wv_intermed = np.append(wv_intermed, wv_ga)
    wv_intermed = np.append(wv_intermed, wv_cluster)
    wv_intermed = np.append(wv_intermed, wv_pi)
    wv_intermed = np.sort(np.unique(wv_intermed))

    # Convert the above into indices for masking over the dataset.
    if(region == "all"):
        wv_intermed_idx = wv_intermed-350
    elif(region == "vis"):
        wv_intermed_idx = wv_intermed-400
    X_train = X_train[:,wv_intermed_idx]

    # Use another genetic algorithm to find the best wavebands out of the narrowed possibilities
    print('\tStarting genetic algorithm...', end=' ')
    wv_consensus = ga(X_train, y_train, trained_pipe, wv_intermed, n_features=max_features_output)
    print('\tDone.')
    return (wv_mi, wv_coeffs, wv_ga, wv_cluster, wv_pi, wv_consensus)


# **The "main" function**

# In[14]:


# Would normally be in the main function, but defined separately for easier testing, debugging, and analysis after running within a Jupyter notebook.
def run():
    # Lists of results that will be compiled at the end into a DataFrame for writing to CSV
    region_list = [] # "all" or "vis"
    gene_list = [] # "bact", "fungi", etc.
    method_list = [] # "mi", "coeffs", etc.
    wv_list = [] # The (int) wavebands selected/considered
    coeff_list = [] # The fitted coefficients for each of the above wavebands (no intercept was calculated)
    penalty_list = [] # The regularization penalty for the model
    ratio_list = [] # The l1-l2 ratio for the model
    rmse_list = [] # The RMSE score for the model
    r2_list = [] # The R2 score for the model
    mae_list = [] # The MAE score for the model

    # Loop over both the whole waveband range and visible light only. (Making the loop in this format in case there are any other regions we want to add later.)
    # Swapped order to vis first, then all, for fail-fast testing
    for starting_region in ("vis", "all"):

        if(starting_region == "all"):
            X_train = X_all_train
            X_test = X_all_test
            X_phoa_train = X_all_phoa_train
            X_phoa_test = X_all_phoa_test
            region_set = wvs
        elif(starting_region == "vis"):
            X_train = X_vis_train
            X_test = X_vis_test
            X_phoa_train = X_vis_phoa_train
            X_phoa_test = X_vis_phoa_test
            region_set = wvs_vis
            
        # Loop over each gene (y value)
        for gene, y_train, y_test in zip(("phoa", "cbblr", "fungi", "bact", "urec"), (phoa_train, cbblr_train, fungi_train, bact_train, urec_train), (phoa_test, cbblr_test, fungi_test, bact_test, urec_test)):
    
            print("Starting ", gene, "...", sep = "")

            # Build a baseline model on the entire region of consideration (no waveband selection methods used) for comparison, and record results
            if(gene == "phoa"):
                baseline_preds, baseline_model = baseline(X_phoa_train, y_train, X_phoa_test)
            else:
                baseline_preds, baseline_model = baseline(X_train, y_train, X_test)
            model = baseline_model.best_estimator_['elastic_net']
            penalty = model.alpha
            ratio = model.l1_ratio
            coeffs = model.coef_
            rmse = root_mean_squared_error(y_test, baseline_preds) * -1
            r2 = r2_score(y_test, baseline_preds)
            mae = mean_absolute_error(y_test, baseline_preds)

            # Record each waveband/coeff separately, in tidy format for easier analysis
            for i, wv in enumerate(region_set):
                region_list.append(starting_region)
                gene_list.append(gene)
                method_list.append("baseline")
                wv_list.append(wv)
                coeff_list.append(coeffs[i])
                penalty_list.append(penalty)
                ratio_list.append(ratio)
                rmse_list.append(rmse)
                r2_list.append(r2)
                mae_list.append(mae)
            # Pickle the model
            model_path = root + '//cache/' + starting_region + "_" + gene + "_baseline.pickle"
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)

            # Where the main calculations happen. Runs each method separately, and then finds the consensus of all of them.
            # The phoa special case is due to a different train/test split than the rest because of some NANs.
            if(gene == "phoa"):
                wv_mi, wv_coeffs, wv_ga, wv_cluster, wv_pi, wv_consensus = consensus(X_phoa_train, y_train, starting_region)
            else:
                wv_mi, wv_coeffs, wv_ga, wv_cluster, wv_pi, wv_consensus = consensus(X_train, y_train, starting_region)
            
            for method, wv_set in zip(("mi", "coeffs", "ga", "cluster", "pi", "consensus"), (wv_mi, wv_coeffs, wv_ga, wv_cluster, wv_pi, wv_consensus)):
                
                # Build a new elastic net model for validation on this subset of wavebands
                if(starting_region == "all"):
                    wv_set_idx = wv_set-350
                elif(starting_region == "vis"):
                    wv_set_idx = wv_set-400
                validator = GridSearchCV(estimator=pipe_elastic_net, param_grid=PARAM_GRID, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv_5_0, error_score='raise')
                # Like above, special case for phoa
                if(gene == "phoa"):
                    validator.fit(X_phoa_train[:,wv_set_idx], y_train)
                    preds = validator.predict(X_phoa_test[:,wv_set_idx])
                else:
                    validator.fit(X_train[:,wv_set_idx], y_train)
                    preds = validator.predict(X_test[:,wv_set_idx])
                model = validator.best_estimator_['elastic_net']
                penalty = model.alpha
                ratio = model.l1_ratio
                coeffs = model.coef_
                rmse = root_mean_squared_error(y_test, preds) * -1
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)

                for i, wv in enumerate(wv_set):
                    region_list.append(starting_region)
                    gene_list.append(gene)
                    method_list.append(method)
                    wv_list.append(wv)
                    coeff_list.append(coeffs[i])
                    penalty_list.append(penalty)
                    ratio_list.append(ratio)
                    rmse_list.append(rmse)
                    r2_list.append(r2)
                    mae_list.append(mae)
                model_path = root + '//cache/' + starting_region + "_" + gene + "_" + method + ".pickle"
                with open(model_path, 'wb') as file:
                    pickle.dump(model, file)
                    
            print("Finished ", gene, ".", sep = "")

    # Compile the results into a single DataFrame and write it to a CSV
    # The format will make for easier analysis later.
    print("Compiling and writing results to CSV...", end = " ")
    col_names = ['gene', 'method', 'wv', 'rmse']
    results = pd.DataFrame(columns = col_names)
    results['region'] = region_list
    results['gene'] = gene_list
    results['method'] = method_list
    results['wv'] = wv_list
    results['coeff'] = coeff_list
    results['penalty'] = penalty_list
    results['ratio'] = ratio_list
    results['rmse'] = rmse_list
    results['r2'] = r2_list
    results['mae'] = mae_list
    results.to_csv(root + '//results/results.csv', index=False)
    print("Done.")


# In[15]:


run()

