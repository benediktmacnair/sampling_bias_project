import argparse
import numpy as np
import random
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

def args_parser():
    parser = argparse.ArgumentParser(description="A simple example script.")
    parser.add_argument('--cov_matrix', type=str)
    parser.add_argument('--mean', type=str)
    parser.add_argument('--n', type=int)

    return parser.parse_args()


class sample:
    def __init__(self, n_sample=1000, n_gauss=2, bad_ratio=0.2, mean=None, cov=None):
        # self.mean = mean
        # self.cov = cov
        self.n_sample = int(n_sample)
        self.n_gauss = int(n_gauss)
        self.bad_ratio = bad_ratio
        


        self.g_mean_acc = []
        self.g_cov_acc = []
        self.g_weight_acc = []
        self.g_sample_acc = []
        # self.g_n_sample = []

        self.b_mean_acc = []
        self.b_cov_acc = []
        self.b_weight_acc = []
        self.b_sample_acc = []
        # self.b_n_sample = []


        self.n_good_sample = int(self.n_sample *(1-self.bad_ratio))
        self.n_bad_sample = int(self.n_sample *(self.bad_ratio))
        self.output_sample = np.array([])


    def random_gaus_params(self):
        # n_params shows how many params are in the dataset. For first phase, we set it to 2
        mean = [random.random(), random.random()]
        var_elem1 = random.uniform(0.1,5)
        var_elem2 = random.uniform(0.1,5)
        cov_elem = random.uniform(-1,1)
        cov_random = [[var_elem1, cov_elem], [cov_elem , var_elem2]]

        return mean, cov_random
    
    def gauss_generate(self, n_sample, cov_acc, mean_acc):
        mean, cov = self.random_gaus_params()

        # size of noise is set as 0.3 from n_sample
        # pdf = mvn(mean=mean, cov=cov).pdf(samples)

        # self.weight_acc.append(weights)
        cov_acc.append(cov)
        mean_acc.append(mean)

        pass

    def generate(self, mean_acc, cov_acc):
        avg_mean = np.mean(self.g_mean_acc, axis=0)
        avg_cov = np.mean(self.g_cov_acc, axis=0)

        gauss = mvn(mean=avg_mean, cov=avg_cov)
        samples = gauss.rvs(size=self.n_sample)
        noise = np.random.normal(loc=0, scale=0.1, size=(int(0.3*self.n_sample),2))
        
        mixed = np.concatenate((samples,noise), axis=0)
        return mixed

    
    def iteration(self):

        for iter in range (self.n_gauss):

            self.gauss_generate(self.n_good_sample, self.g_cov_acc, self.g_mean_acc)
            self.gauss_generate(self.n_bad_sample, self.b_cov_acc, self.b_mean_acc)

        good_samples = self.generate(self.g_mean_acc, self.g_cov_acc)
        good_samples = np.hstack((good_samples, np.zeros((good_samples.shape[0], 1))))  

        bad_samples = self.generate(self.b_mean_acc, self.b_cov_acc)
        bad_samples = np.hstack((bad_samples, np.ones((bad_samples.shape[0], 1))))  
        
        self.output_sample = np.concatenate((good_samples, bad_samples), axis=0)

            



def main():

    test = sample(n_gauss=5)
    test.iteration()

if __name__ == "__main__": 
    main()