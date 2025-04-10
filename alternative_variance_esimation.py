import numpy as np

def psi_e(lam):
    return -lam - np.log(1 - lam)
    
def psi_p(lam):
    return np.exp(lam) - lam - 1
        
### MP class
class UnivariateMaurerPontil:
    def __init__(self, alpha = 0.05):
        self.alpha = alpha
        self.X = None

    def __call__(self, X):

        if self.X is None:
            self.X = np.array([X])
        else:   
            self.X = np.concatenate([self.X, np.array([X])])

    def get_center_plus_radius(self):
        return np.std(self.X) + np.sqrt(2*np.log(1/self.alpha)/(len(self.X)-1))
    
    def get_center_minus_radius(self):
        sol = np.std(self.X) - np.sqrt(2*np.log(1/self.alpha)/(len(self.X)-1))
        return np.max( (sol, 0))
    
    def get_center_plus_radius_sharper(self):
        return np.sqrt( np.var(self.X) + np.log(1/self.alpha) / (2*(len(self.X)-1)) ) + np.sqrt( np.log(1/self.alpha) / (2*(len(self.X)-1)))
    
    def get_center_minus_radius_sharper(self):
        sol = np.sqrt( np.var(self.X) - np.log(1/self.alpha) / (2*(len(self.X)-1)) ) - np.sqrt( np.log(1/self.alpha) / (2*(len(self.X)-1)))
        return np.max( (sol, 0)) 
    
    def get_center(self):
        return np.std(self.X)


### Decoupled double empirical Bernstein
class UnivariateDecoupledEmpiricalBernstein:

    def __init__(self, max_rounds, alpha = 0.05, c1 = 0.5, c2 = 0.25, CS=False, tilde_alpha = 0.05, tilde_c1 = 0.5, tilde_c2 = 0.25, tilde_CS=False):
        self.max_rounds = max_rounds
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.CS = CS
        self.tilde_alpha = tilde_alpha
        self.tilde_c1 = tilde_c1
        self.tilde_c2 = tilde_c2
        self.tilde_CS = tilde_CS

        self.t = 0

        self.sumvarhat = self.c2 
        self.summuhat = self.c1
        self.aux = np.sqrt(2*np.log(1/self.alpha))
        self.part1 = 0
        self.part2 = np.log(1/self.alpha)
        self.sum_lambdas = 0
        self.sum_bernstein_center = 0

        self.sum_tilde_varhat = self.tilde_c2
        self.sum_tilde_muhat = self.tilde_c1
        self.tilde_aux = np.sqrt(2*np.log(1/self.tilde_alpha))
        self.tilde_part1 = 0
        self.tilde_part2 = np.log(1/self.tilde_alpha)
        self.sum_tilde_lambdas = 0
        self.sum_tilde_bernstein_center = 0
        
        
    def __call__(self, Y):
        
        X = Y**2
        radius = (X - self.summuhat/(self.t+1))**2
        if self.CS:
            self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * np.log(self.t+2)), self.c1) 
        else:
            self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * self.max_rounds / (self.t + 1)), self.c1)
        self.sum_lambdas += self.lambda_b
        # Center
        self.sum_bernstein_center += self.lambda_b * X
        self.bernstein_center = self.sum_bernstein_center / self.sum_lambdas
        # Radius
        self.part1 += radius*psi_e(self.lambda_b) 
        self.bernstein_radius = (self.part1+self.part2)/self.sum_lambdas

        tilde_radius = (Y - self.sum_tilde_muhat/(self.t+1))**2
        if self.tilde_CS:
            self.tilde_lambda_b = np.minimum(self.tilde_aux / np.sqrt(self.sum_tilde_varhat * np.log(self.t+2)), self.tilde_c1)
        else:
            self.tilde_lambda_b = np.minimum(self.tilde_aux / np.sqrt(self.sum_tilde_varhat * self.max_rounds / (self.t + 1)), self.tilde_c1)
        self.sum_tilde_lambdas += self.tilde_lambda_b
        # Tilde center
        self.sum_tilde_bernstein_center += self.tilde_lambda_b * Y
        self.tilde_bernstein_center = self.sum_tilde_bernstein_center / self.sum_tilde_lambdas
        # Tilde radius
        self.tilde_part1 += tilde_radius*psi_e(self.tilde_lambda_b) 
        self.tilde_bernstein_radius = (self.tilde_part1+self.tilde_part2)/self.sum_tilde_lambdas

        # Update auxiliary variables
        self.summuhat += X
        self.sumvarhat += radius
        self.sum_tilde_muhat += Y
        self.sum_tilde_varhat += tilde_radius

        
        
        self.t += 1

    def get_upper_bound(self):
        u1 = self.bernstein_center + self.bernstein_radius
        l2 = self.tilde_bernstein_center - self.tilde_bernstein_radius
        return np.sqrt(u1 - l2**2)
    
    def get_lower_bound(self):
        l1 = self.bernstein_center - self.bernstein_radius
        u2 = self.tilde_bernstein_center + self.tilde_bernstein_radius
        return np.sqrt(l1 - u2**2)

    def get_center(self):
        return np.sqrt(self.bernstein_center - self.tilde_bernstein_center)
    
    def get_second_moment(self):
        return self.bernstein_center
    
    def get_mean(self):
        return self.tilde_bernstein_center  


### This class is like EBLB but with a different $\widehat\sigma_i^2$, which is centered at $\hat\mu_i$ instead of $\bar\mu_i$.
class EBLBmod:
    def __init__(self, max_rounds, alpha = 0.05, tilde_alpha = 0.05,
                 c1 = 0.5, c2 = 0.25**2, c3=0.25, c4=0.5, c5=2,
                 CS=False, tilde_CS=False):
        self.max_rounds = max_rounds
        self.CS = CS
        self.tilde_CS = tilde_CS
        self.alpha = alpha
        self.tilde_alpha = tilde_alpha

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        

        self.t = 0
        self.sumvarhat = self.c2 
        self.summuhat = self.c3
        self.tilde_sumvarhat = self.c3
        self.tilde_summuhat = self.c4
        self.aux = np.sqrt(2*np.log(1 / self.alpha))
        self.part1 = 0
        self.part2 = np.log(1/self.alpha)
        self.sum_lambdas = 0
        
        self.bernstein_center = None
        self.bernstein_radius = None
        self.sum_bernstein_center = 0
        
        
        self.tilde_aux = np.sqrt(2*np.log(2 / self.tilde_alpha))
        
        self.tilde_sum_bernstein_center = 0
        self.tilde_bernstein_center = 0
        self.sum_psi_tilde_lambdas = 0
        self.sum_tilde_lambdas = 0

        self.dt = None
        self.rt = None
        self.tilde_at = None
        self.tilde_bt = None
        self.tilde_ct = None
        self.sum_at = 0
        self.sum_bt = 0
        self.sum_ct = 0
        self.at = None
        self.bt = None
        self.ct = None



        
    def __call__(self, Y):

        # Empirical Bernstein for the variance
        X = (Y - self.tilde_bernstein_center)**2
        radius = (X - self.summuhat/(self.t+1))**2

        varhat = self.tilde_sumvarhat / (self.t+1) # radius 
        if self.sum_tilde_lambdas > 0:
            threshold = (np.log(2/self.tilde_alpha) + varhat * self.sum_psi_tilde_lambdas)/self.sum_tilde_lambdas
            condition_upsilon = threshold < 1
            if condition_upsilon:
                if self.CS:
                    self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * np.log(self.t+2)), self.c1) 
                else:
                    self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * self.max_rounds / (self.t+1)), self.c1)
            else:
                self.lambda_b = 0
        else:
            self.lambda_b = 0
        self.sum_lambdas += self.lambda_b
        
        # Center
        self.sum_bernstein_center += self.lambda_b * X
        # Radius
        self.part1 += radius*psi_e(self.lambda_b)
        # Both
        if self.sum_lambdas > 0:
            self.bernstein_center = self.sum_bernstein_center / self.sum_lambdas
            self.bernstein_radius = (self.part1+self.part2)/self.sum_lambdas

        # Extra term due to variance overestimation
        if self.t > 0:

            self.tilde_at = self.sum_psi_tilde_lambdas**2 / self.sum_tilde_lambdas**2
            self.tilde_bt = 2*np.log(2/self.tilde_alpha)*self.sum_psi_tilde_lambdas / self.sum_tilde_lambdas**2
            self.tilde_ct = np.log(2/self.tilde_alpha)**2 / self.sum_tilde_lambdas**2

            self.dt = self.bernstein_center
            self.rt = self.bernstein_radius
            
            if self.lambda_b > 0:      
                self.sum_at += self.tilde_at * self.lambda_b
                self.at = self.sum_at / self.sum_lambdas
                self.sum_bt += self.tilde_bt * self.lambda_b
                self.bt = 1 + self.sum_bt / self.sum_lambdas
                self.sum_ct += self.tilde_ct * self.lambda_b
                self.ct = self.sum_ct / self.sum_lambdas
            
    
        # Update empirical Bennett
        # Tilde lambdas
        tilde_radius = (Y - self.tilde_summuhat/(self.t+1))**2
        if self.tilde_CS:
            self.tilde_lambda_b = np.minimum(self.tilde_aux / np.sqrt(self.tilde_sumvarhat * np.log(self.t+2) ), self.c5)
        else:
            self.tilde_lambda_b = np.minimum(self.tilde_aux / np.sqrt(self.tilde_sumvarhat * self.max_rounds / (self.t+1)), self.c5)
        self.sum_tilde_lambdas += self.tilde_lambda_b
        self.sum_psi_tilde_lambdas += psi_p(self.tilde_lambda_b)
        # Tilde center
        self.tilde_sum_bernstein_center += self.tilde_lambda_b * Y
        self.tilde_bernstein_center = self.tilde_sum_bernstein_center / self.sum_tilde_lambdas

        # Update auxiliary variables
        self.summuhat += X
        self.sumvarhat += radius
        self.tilde_summuhat += Y
        self.tilde_sumvarhat += tilde_radius
        self.t += 1
        
    def get_center_minus_radius(self):
        c = self.dt - self.ct - self.rt
        if c < 0:
            return 0
        else:
            sol = (-self.bt + np.sqrt(self.bt**2 + 4*self.at*c)) / (2*self.at)
            return np.sqrt(sol)
    
    def get_center(self):
        return np.sqrt(self.bernstein_center)


### This class defines the R 
class UnivariateEmpiricalBernsteinBennettLowerBound:
    def __init__(self, max_rounds, alpha = 0.05, c1 = 0.5, c2 = 0.25, c3=0.25, c4=0.5, c5=2, 
                 tilde_alpha = 0.05, CS=False, tilde_CS=False):
        self.max_rounds = max_rounds
        self.CS = CS
        self.alpha = alpha
        self.tilde_CS = tilde_CS
        self.tilde_alpha = tilde_alpha

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5

        self.sumvarhat = self.c2 
        self.summuhat = self.c3
        self.tilde_sumvarhat = self.c3
        self.tilde_summuhat = self.c4

        self.t = 0
        self.aux = np.sqrt(2*np.log(1 / self.alpha))
        self.part1 = 0
        self.part2 = np.log(1/self.alpha)
        self.sum_lambdas = 0
        self.sum_bernstein_center = 0
        
        self.tilde_aux = np.sqrt(2*np.log(2 / self.tilde_alpha))
        self.tilde_sum_bernstein_center = 0
        self.tilde_bernstein_center = 0
        self.sum_psi_tilde_lambdas = 0
        self.sum_tilde_lambdas = 0

        self.dt = None
        self.rt = None
        self.tilde_at = None
        self.tilde_bt = None
        self.tilde_ct = None
        self.sum_at = 0
        self.sum_bt = 0
        self.sum_ct = 0
        self.at = None
        self.bt = None
        self.ct = None

        self.upsilon_count = 0


        
    def __call__(self, Y):

        # Empirical Bernstein for the variance
        X = (Y - self.tilde_bernstein_center)**2
        radius = (X - self.summuhat/(self.t+1))**2
        
        if self.CS:
            self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * np.log(self.t+2)), self.c1) 
        else:
            self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * self.max_rounds / (self.t + 1)), self.c1)
        self.sum_lambdas += self.lambda_b
        

        # Center
        self.sum_bernstein_center += self.lambda_b * X
        self.bernstein_center = self.sum_bernstein_center / self.sum_lambdas
        # Radius
        self.part1 += radius*psi_e(self.lambda_b)
        self.bernstein_radius = (self.part1+self.part2)/self.sum_lambdas

        # Extra term due to variance overestimation CHECK
        varhat = self.tilde_sumvarhat / (self.t+2) # radius 
        if self.t > 0:
            threshold = (np.log(2/self.tilde_alpha) + varhat * self.sum_psi_tilde_lambdas)/self.sum_tilde_lambdas
            condition_upsilon = (threshold < 1)

            self.dt = self.bernstein_center
            self.rt = self.bernstein_radius
            self.tilde_at = self.sum_psi_tilde_lambdas**2 / self.sum_tilde_lambdas**2
            self.tilde_bt = 2*np.log(2/self.tilde_alpha)*self.sum_psi_tilde_lambdas / self.sum_tilde_lambdas**2
            self.tilde_ct = np.log(2/self.tilde_alpha)**2 / self.sum_tilde_lambdas**2
            if condition_upsilon: 
                self.upsilon_count += 1
                self.sum_at += self.tilde_at * self.lambda_b
                self.at = self.sum_at / self.sum_lambdas
                self.sum_bt += self.tilde_bt * self.lambda_b
                self.bt = 1 + self.sum_bt / self.sum_lambdas
                self.sum_ct += self.tilde_ct * self.lambda_b
                self.ct = self.sum_ct / self.sum_lambdas
            else:
                self.sum_ct += self.lambda_b
                self.ct = self.sum_ct / self.sum_lambdas
        else:
            self.sum_ct += self.lambda_b
            self.ct = self.sum_ct / self.sum_lambdas
    
        # Update empirical Bennett
        # Tilde lambdas
        tilde_radius = (Y - self.tilde_summuhat/(self.t+1))**2
        if self.tilde_CS:
            self.tilde_lambda_b = np.minimum(self.tilde_aux / np.sqrt(self.tilde_sumvarhat * np.log(self.t+2) ), self.c5)
        else:
            self.tilde_lambda_b = np.minimum(self.tilde_aux / np.sqrt(self.tilde_sumvarhat * self.max_rounds / (self.t + 1)), self.c5)
        self.sum_tilde_lambdas += self.tilde_lambda_b
        self.sum_psi_tilde_lambdas += psi_p(self.tilde_lambda_b)
        # Tilde center
        self.tilde_sum_bernstein_center += self.tilde_lambda_b * Y
        self.tilde_bernstein_center = self.tilde_sum_bernstein_center / self.sum_tilde_lambdas
        

        # Update auxiliary variables
        self.summuhat += X
        self.sumvarhat += radius
        self.tilde_summuhat += Y
        self.tilde_sumvarhat += tilde_radius
        self.t += 1
        
    def get_center_minus_radius(self):
        c = self.dt - self.ct - self.rt
        if c < 0:
            return 0
        else:
            sol = (-self.bt + np.sqrt(self.bt**2 + 4*self.at*c)) / (2*self.at)
            return np.sqrt(sol)
    
    def get_center(self):
        return np.sqrt(self.bernstein_center)