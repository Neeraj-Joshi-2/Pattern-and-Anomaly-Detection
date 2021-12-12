# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:26:57 2021

@author: NEERAJ
"""


import sklearn.datasets as dd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#%%

X, y = dd.make_regression(100, 5)
print("X\n")
print(X)
transformer = FunctionTransformer(np.exp)
new_X = transformer.transform(X)
print()
print("new_X\n")
print(new_X)

#%%
#for x
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)

#%%
#for new_X
X_train2, X_test2, y_train2, y_test2 = train_test_split(new_X, y, test_size=0.2)

print('Shape of X_train2 : ' , X_train2.shape)
print('Shape of y_train2 : ' , y_train2.shape)
print('Shape of X_test2 : ' , X_test2.shape)
print('Shape of y_test2 : ' , y_test2.shape)

#%%
reg = LinearRegression()
#for dataset X
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(y_pred)

#for dataset new_X
reg.fit(X_train2,y_train2)
y_pred2 = reg.predict(X_test2)
print(y_pred2)

#%%
#score

score1 = r2_score(y_test, y_pred)
print('Score for data X using exp : ' , score1)
score2 = r2_score(y_test2, y_pred2)
print('Score for data X_new using exp : ' , score2)

#%%
# For sigmoid
transformer2 = FunctionTransformer(np.tanh)
new_X = transformer.transform(X)
print(new_X)
#%%
X_train3, X_test3, y_train3, y_test3 = train_test_split(new_X, y, test_size=0.2)


#%%
reg.fit(X_train3,y_train3)
y_pred3 = reg.predict(X_test3)
print(y_pred3)

#%%
#score
score3 = r2_score(y_test3, y_pred3)
print('Score using tanh : ' , score3)

#%%
import matplotlib.pyplot as plt
import numpy as np


def plot_gpr_samples(gpr_model, n_samples, ax):

    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-3, 3])



rng = np.random.RandomState(4)
X_train = rng.uniform(0, 5, 10).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
n_samples = 5


### RBF Kernel

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

fig.suptitle("Radial Basis Function kernel", fontsize=18)
plt.tight_layout()


print(f"Kernel parameters before fit:\n{kernel})")
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)



#%%
### Rational Quadratic Kernel

from sklearn.gaussian_process.kernels import RationalQuadratic

kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

fig.suptitle("Rational Quadratic kernel", fontsize=18)
plt.tight_layout()

print(f"Kernel parameters before fit:\n{kernel})")
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)