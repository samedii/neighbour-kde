import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.integrate

#weight based on distance from interesting point

#extra-/inter-polate points (move them)

#multivariate kernel - handle missing data - this is not multivariate at the moment... we are doing conditional stuff

#combine with local regression? some parameters would be estimated with fixed weights?

#KDE of MCMC samples? move around samples to improve fit?

#WAIT WHAT? WE CAN GET WEIGHTS FOR KDE WITH IM? CAN WE? No the truth is unknown

#IM sampler where samples create new distribution?
#Change smoothing parameter based on how well new samples were predicted
#Sample weights decided by IM? Not possible since they are dependent? Can we do a batch at a time and throw away old?
#Would detailed balance be satisfied if we keep samples?
#Deterministic sampling (batches?) - can we do that with KDE?
#
# Possible steps
#   1. Optimize and get maximum
#   2. Laplace approximation
#   3. Run MCMC - n samples (prefer IM and sampling deterministically?)
#   4. Calculate weights (optional, only possible with IM)
#   5. Select smoothing parameter of KDE with CV (and kernel?)
#   6. Go to step 3 and use KDE as proposal (keep old samples - can we recalibrate weights? detailed balance satisfied?)
#   Repeat until new batch of samples is close enough to expected
#
#   Can this be done on mini-batch? Sort of?

np.random.seed(1)

len_data = 100
X_data = scipy.stats.uniform.rvs(loc=0, scale=1, size=len_data)*2-1
y_data = scipy.stats.norm.rvs(loc=4*X_data**4 + X_data, scale=1)

#plt.hist(y_data)
#plt.show()

# plt.scatter(X_data, y_data)
# plt.show()

kernel_sigma = 0.1
distance_sigma = 0.1

def pdf(y, X, y_data, X_data, kernel_sigma, distance_sigma):
    yy, yy_data = np.meshgrid(y, y_data)
    yX, yX_data = np.meshgrid(X, X_data)
    weights = scipy.stats.norm.pdf(yX_data, loc=yX, scale=distance_sigma)
    weights = weights/weights.sum(axis=0)
    return (scipy.stats.norm.pdf(yy, loc=yy_data, scale=kernel_sigma)*weights).sum(axis=0)

look_at_X = -2
y = np.linspace(start=-3, stop=3, num=100)
X = np.repeat(look_at_X, 100)
# plt.plot(y, scipy.stats.norm.pdf(y, loc=X*0.5, scale=y_sigma))
# plt.plot(y, pdf(y, X, y_data, X_data, kernel_sigma, distance_sigma))
# plt.show()


mask = np.logical_not(np.identity(len_data, dtype=np.bool))
yloc = np.reshape(np.tile(y_data, (len_data, 1))[mask], (len_data, len_data-1))
ydata = np.transpose(np.tile(y_data, (len_data-1, 1)))
Xloc = np.reshape(np.tile(X_data, (len_data, 1))[mask], (len_data, len_data-1))
Xdata = np.transpose(np.tile(X_data, (len_data-1, 1)))

def leave_one_out_logpdf(kernel_sigma, distance_sigma):
    weights = scipy.stats.norm.pdf(Xdata, loc=Xloc, scale=distance_sigma)
    weights = weights/weights.sum(axis=0)
    return np.log((scipy.stats.norm.pdf(ydata, loc=yloc, scale=kernel_sigma)*weights).sum(axis=0)).sum()

log_sigma0 = np.log([0.1, 0.1])
result = scipy.optimize.minimize(fun=lambda log_sigma: -leave_one_out_logpdf(np.exp(log_sigma)[0], np.exp(log_sigma)[1]), x0=log_sigma0)

print(result)
kernel_sigma = np.exp(result.x[0])
distance_sigma = np.exp(result.x[1])
print('kernel_sigma:', kernel_sigma, ', distance_sigma:', distance_sigma)

# plot a plane
# plt.plot(y, scipy.stats.norm.pdf(y, loc=X*0.5, scale=y_sigma))
# plt.plot(y, pdf(y, X, y_data, X_data, kernel_sigma, distance_sigma))
# plt.show()

# contour
plt.scatter(X_data, y_data)
x = np.linspace(-1, 1, num=100)
y = np.linspace(-3, 6, num=100)
xx, yy = np.meshgrid(x, y)

# print(pdf(yy, xx, y_data, X_data, kernel_sigma, distance_sigma))
plt.contour(x, y,
    pdf(yy, xx, y_data, X_data, kernel_sigma, distance_sigma).reshape(len(x), len(y)))
plt.scatter(X_data, y_data)
plt.show()

