import numpy as np

class ExperimentsParameters():
    def __init__(self):
        
        self.alpha = 0.05
        self.c1 = .5
        self.c2 = 0.25**2
        self.c3 = .25
        self.c4 = 0.5
        self.c5 = 2
        self.n_experiments = 100
        self.sample_size_list = np.arange(100, 5000, 100)
        self.distribution_type_list = ["uniform", "beta", "beta1"]
        self.CS = False
        self.tilde_CS = True
        
    def alpha_proportion(self, max_rounds):
        return 1/2
        return (max_rounds-1)/max_rounds

