import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.integrate

np.random.seed(1)

#weight based on distance from interesting point

#extra-/inter-polate points (move them)

len_y_data = 100
y_data = scipy.stats.norm.rvs(loc=0, scale=1, size=len_y_data)

#plt.hist(y_data)
#plt.show()

sigma = 0.1

def pdf(y, y_data, sigma):
    yy, yy_data = np.meshgrid(y, y_data)
    return scipy.stats.norm.pdf(yy, loc=yy_data, scale=sigma).mean(axis=0)
    #return scipy.stats.norm.logpdf(np.transpose(np.tile(y, (len(y_data), 1))), loc=np.tile(y_data, (len(y), 1)), scale=sigma).mean(axis=1)

y = np.linspace(start=-3, stop=3, num=100)

plt.plot(y, scipy.stats.norm.pdf(y, loc=0, scale=1))
plt.plot(y, pdf(y, y_data, sigma))
plt.show()

mask = np.logical_not(np.identity(len_y_data, dtype=np.bool))
loc = np.reshape(np.tile(y_data, (len_y_data, 1))[mask], (len_y_data, len_y_data-1))
data = np.transpose(np.tile(y_data, (len_y_data-1, 1)))

def leave_one_out_logpdf(sigma):
    return np.log(scipy.stats.norm.pdf(data, loc=loc, scale=sigma).mean(axis=0)).sum()

sigma0 = 0.1
result = scipy.optimize.minimize(fun=lambda sigma: -leave_one_out_logpdf(sigma), x0=sigma0)

print(result)
sigma = result.x[0]

plt.plot(y, scipy.stats.norm.pdf(y, loc=0, scale=1))
plt.plot(y, pdf(y, y_data, sigma))
plt.show()



