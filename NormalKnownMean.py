import numpy as np
from scipy.stats import invgamma, norm
from scipy.integrate import dblquad
import matplotlib.pyplot as plt 

class NormalKnownMean:
    
    def __init__(self, alpha: int, beta: int, known_mean: int, num_bins: int, observation: None):
        self.alpha = alpha
        self.beta = beta
        self.mean = known_mean
        self.num_bins = num_bins
        self.observation = observation

    def get_alpha(self) -> int:
        return self.alpha
    
    def set_alpha(self, alpha: int) -> None:
        self.alpha = alpha

    def get_beta(self) -> int:
        return self.beta
    
    def set_beta(self, beta: int) -> None:
        self.beta = beta

    def get_mean(self) -> int:
        return self.mean
    
    def get_bins(self) -> int:
        return self.num_bins
    
    def set_bins(self, num_bins: int) -> None:
        self.num_bins = num_bins

    def get_observation(self) -> int:
        return self.observation

    def set_observation(self, data: float) -> None:
        self.observation = data
    
    def x_bins(self): 
        numBins = self.get_bins()
        mu = self.get_alpha()
        std = self.get_beta()
        norm_dist = norm(loc=mu, scale=std)
        max_abs_boundary = max(abs(norm_dist.ppf(0.00001)), abs(norm_dist.ppf(0.99999)))
        lower_boundary = max(-max_abs_boundary, 0)
        bin_edges = np.linspace(lower_boundary, max_abs_boundary, numBins+1)
        return bin_edges
    
    def y_bins(self): 
        numBins = self.get_bins()
        mu = self.get_alpha()
        std = self.get_beta()
        norm_dist = norm(loc=mu, scale=std)
        max_abs_boundary = max(abs(norm_dist.ppf(0.00001)), abs(norm_dist.ppf(0.99999)))
        lower_boundary = max(-max_abs_boundary, 0)
        bin_edges = np.linspace(lower_boundary, max_abs_boundary, numBins+1)
        return bin_edges
    
    def prior_pdf(self,x): # ASSUMING THIS IS THE CORRECT PRIOR PDF
        return norm.pdf(x, self.get_alpha(), scale=np.sqrt(self.get_beta()))
    
    def likelihood(self, x, y):
        sigma = np.sqrt(self.get_beta())
        return (1 / (np.sqrt(2* np.pi * self.get_beta()))) * (np.exp(-0.5 * ((y - x) / sigma) ** 2)) 
    
    def discrete_likelihood(self):
        x_bin = self.x_bins()
        y_bin = self.y_bins()
        f = lambda x, y: self.prior_pdf(x) * self.likelihood(x, y)
        result_matrix = np.zeros((len(x_bin) - 1, len(y_bin) - 1))
        for i in range(len(x_bin) - 1):
            for j in range(len(y_bin) - 1):
                result = dblquad(f, y_bin[j], y_bin[j+1], x_bin[i], x_bin[i+1])
                result_matrix[i, j] = result[0]
        return result_matrix
    
    def bayes_theorem_formula(self): 
        matrix = self.discrete_likelihood()
        col_sums = matrix.sum(axis=0)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        matrix /= col_sums
        return matrix
    
    def observation_data(self):
        data = 1
        post = self.conujugate_prior(data)
        data = np.abs(np.random.normal(post[0], post[1], 1))
        return data
    
    def post_distribution(self):
        data = 1
        post = self.conujugate_prior(data)
        return post

    def conujugate_prior(self, data):
        return ((self.get_alpha() + data/2), self.get_beta() + (((data - self.get_mean())**2) / 2))
    
    def discrete_posterior(self): # THE SUM OF THIS MATRIX = 1
        if self.get_observation() is None:
            data = self.observation_data()
            self.set_observation(data)
        else:
            data = self.get_observation()
        ybin = self.y_bins()
        column_matrix = np.zeros((len(ybin) - 1, 1))
        for i in range(len(ybin) - 1):
            if data > ybin[i] and data <= ybin[i+1]:
                column_matrix[i] = [1.0]
                break
        bayes_matrix = self.bayes_theorem_formula()
        discrete_post_matrix = np.dot(bayes_matrix, column_matrix)
        print(np.sum(discrete_post_matrix))
        return discrete_post_matrix
    
    def draw_post(self): 
        data = self.discrete_posterior() # Obtains the column vector of discrete posterior values
        bins = self.y_bins() # Creates the range of bins
        bin_widths = np.diff(bins)
        data = data / bin_widths.reshape(-1, 1)
        domain = np.linspace(0, bins[len(bins)-1], 1000) # Domain space for true posterior distribution
        post = self.post_distribution() # Creates the true posterior distribution
        fig, axes = plt.subplots()
        axes.stairs(np.ravel(data), bins, label='Discrete Posterior') # Creates the stairs plot using the data and bins excluding the bins that go to +- infinity
        axes.plot(domain, invgamma.pdf(domain, post[0], post[1]), label = 'Conjugate Prior Posterior') # Plots the conjugate prior over the discrete 
        axes.plot(bins[:-1],np.ravel(data), 'o--', color='grey', alpha=0.6)
        axes.axvline(self.get_observation(), color = 'red', linestyle='--', label = 'Observation = {:.6g}'.format(float(self.get_observation()))) # Plot the observation value
        axes.axvline(self.get_beta(), color = 'green', linestyle='--', label = 'Prior Variance = ' + str(self.get_beta())) # Plot the prior mean to see the difference between it and observation
        y_ticks = (np.arange(0, 1.1, 0.1))
        y_ticklabels = ['{:.1f}'.format(tick) for tick in y_ticks]
        axes.set_yticks(y_ticks)
        axes.set_yticklabels(y_ticklabels, fontsize=16)
        axes.set_title("Step Graph of Discrete Posterior with " +  str(self.get_bins()) + " bins", fontsize=16)
        axes.set_xlabel('Variance', fontsize=16)
        axes.set_ylabel('Probability', fontsize=16)
        axes.legend(fontsize=16)
        plt.show()
        

normalKnownMean = NormalKnownMean(0, 1, 1, 5, None)



# print(normalKnownMean.discrete_posterior())
# print(normalKnownMean.y_bins())
normalKnownMean.draw_post()

