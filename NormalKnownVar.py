import numpy as np
from scipy.stats import norm
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

class NormalKnownVar:
    
    def __init__(self, mu: int, prior_std: int, known_std: int, numBins: int, observation: float):
        self.mu = mu
        self.prior_std = prior_std
        self.known_std = known_std
        self.observation = observation
        self.numBins = numBins

    def get_mu(self) -> int:
        return self.mu
    
    def set_mu(self, mu : int) -> None:
        self.mu = mu
    
    def get_prior_std(self) -> int:
        return self.prior_std
    
    def set_prior_std(self, prior_std) -> None:
        self.prior_std = prior_std

    def get_known_std(self) -> int:
        return self.known_std
    
    def set_known_std(self, known_std: int) -> None:
        self.known_std = known_std

    def get_observation(self) -> int:
        return self.observation

    def set_observation(self, data: float) -> None:
        self.observation = data

    def get_bins(self) -> int:
        return self.numBins
    
    def set_bins(self, numBins: int) -> None:
        self.numBins = numBins
    
    def x_bins(self): 
        numBins = self.get_bins()
        norm_dist = norm(loc=self.get_mu(), scale=self.get_prior_std())
        bin_edges = norm_dist.ppf(np.linspace(0.001, 0.999, numBins + 1))
    
        # Returns the edges of the bins
        return bin_edges
    

    def y_bins(self): 
        numBins = self.get_bins()
        norm_dist = norm(loc=self.get_mu(), scale=self.get_prior_std())
        bin_edges = norm_dist.ppf(np.linspace(0.001, 0.999, numBins + 1))
    
        # Returns the edges of the bins
        return bin_edges

    def prior_pdf(self, x):
        return norm.pdf(x, loc=self.get_mu(), scale=np.sqrt(self.get_prior_std())) 

    def likelihood(self, x, y):
        sigma = np.sqrt(self.get_known_std())
        return (1 / (np.sqrt(2* np.pi * self.get_known_std()))) * (np.exp(-0.5 * ((y - x) / sigma) ** 2)) 
    
    def discrete_likelihood(self): 
        x_bin = self.x_bins() # Variable to store the bins
        y_bin = self.y_bins() # Variable to store the bins
        f = lambda x, y: self.prior_pdf(x) * self.likelihood(x, y) # Creates the Integrand.
        result_matrix = np.zeros((len(x_bin) - 1, len(y_bin) - 1)) # Creates a 3x3 matrix
        for i in range(len(x_bin) - 1): # Will start with the x bin jumping to each possible y bin
            for j in range(len(y_bin) - 1): # Loops over each y bin 
                result = dblquad(f, y_bin[j], y_bin[j+1],x_bin[i], x_bin[i+1]) #Computes the double integral
                result_matrix[i, j] = result[0] # Adds the result to the matrix
        return result_matrix # Returns the matrix
    
    def bayes_theorem_formula(self):    # Divides each element in the matrix by the sum of its column
        matrix = self.discrete_likelihood()
        col_sums = matrix.sum(axis=0)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        matrix /= col_sums
        return matrix
    
    def observation_data(self): # Uses the Conjugate Prior formula to obtain a Posterior Distribution
        denominator = (1.0/self.get_prior_std()**2 + 1/self.get_known_std()**2)
        data = 1  # Takes 1 observation
        post = self.conjugate_prior_formula(data, denominator)
        data = np.random.normal(post[0], post[1], 1)
        return data # Returns our single observation
    
    def post_distribution(self):
        denominator = (1.0/self.get_prior_std()**2 + 1/self.get_known_std()**2)
        data = 1 
        post = self.conjugate_prior_formula(data, denominator)
        return post # Returns the posterior distribution when taking 1 observation

    def conjugate_prior_formula(self, data, denominator): # Conjugate Prior formula
        return  ((self.get_mu() / self.get_prior_std() + data / self.get_known_std()**2) / denominator, np.sqrt(1.0 / denominator)) 


    def discrete_posterior(self): 
        if self.get_observation() is None:
            data = self.observation_data()
            self.set_observation(data)
        else:
            data = self.get_observation()
        ybin = self.y_bins()
        column_matrix = np.zeros((len(ybin) - 1, 1))
        for i in range(len(ybin) - 1):
            if data > ybin[i] and data <= ybin[i+1]: # Checks where our observation lies within our bin space
                column_matrix[i] = [1.0] # Changes the column matrix value where the observation lies to 1
                break
        bayes_matrix = self.bayes_theorem_formula()
        discrete_post_matrix = np.dot(bayes_matrix, column_matrix) # Computes the discrete posterior column matrix
        return discrete_post_matrix
    
    def draw_post(self): 
        data = self.discrete_posterior() # Obtains the column vector of discrete posterior values
        bins = self.y_bins() # Creates the range of bins
        bin_widths = np.diff(bins)
        data = data / bin_widths.reshape(-1, 1)
        domain = np.linspace(self.get_mu() - 15, self.get_mu() + 15, 1000) # Domain space for true posterior distribution
        post = self.post_distribution() # Creates the true posterior distribution
        fig, axes = plt.subplots()
        axes.stairs(np.ravel(data), bins, label='Discrete Posterior') # Creates the stairs plot using the data and bins.
        axes.plot(domain, norm.pdf(domain, post[0], post[1]), label = 'Conjugate Prior Posterior') # Plots the conjugate prior over the discrete 
        axes.plot(bins[:-1],np.ravel(data), 'o--', color='grey', alpha=0.6)
        axes.axvline(self.get_observation(), color = 'red', linestyle='--', label = 'Observation = {:.6g}'.format(float(self.get_observation()))) # Plot the observation value
        axes.axvline(self.get_mu(), color = 'green', linestyle='--', label = 'Prior Mean = ' + str(self.get_mu())) # Plot the prior mean to see the difference between it and observation
        axes.set_xlim(self.get_mu() - 5, self.get_mu() + 15)
        y_ticks = (np.arange(0, 0.9, 0.1))
        y_ticklabels = ['{:.1f}'.format(tick) for tick in y_ticks]
        axes.set_yticks(y_ticks)
        axes.set_yticklabels(y_ticklabels, fontsize=16)

        axes.set_xlabel('Mean', fontsize=16)
        axes.set_ylabel('Probability', fontsize=16)
        axes.set_title("Step Graph of Discrete Posterior with " +  str(self.get_bins()) + " bins", fontsize=16)
        axes.legend(fontsize=16)
        plt.show()

        

normalKnownVar = NormalKnownVar(0, 1, 1, 5, None)
# print(normalKnownVar.x_bins())
normalKnownVar.draw_post()



