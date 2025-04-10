import numpy as np
from tqdm import tqdm

from alternative_variance_esimation import UnivariateMaurerPontil, UnivariateDecoupledEmpiricalBernstein, UnivariateEmpiricalBernsteinBennettLowerBound
from variance_estimation import EBLB, EBUB
from experiment_params import ExperimentsParameters

if __name__ == "__main__":

    np.random.seed(0)

    params = ExperimentsParameters()
    alpha = params.alpha
    c1 = params.c1
    c2 = params.c2
    c3 = params.c3
    c4 = params.c4
    c5 = params.c5
    n_experiments = params.n_experiments
    sample_size_list = params.sample_size_list
    distribution_type_list = params.distribution_type_list
    alpha_proportion = params.alpha_proportion
    CS = params.CS
    tilde_CS = params.tilde_CS
    

    T_reshaped = np.zeros((n_experiments, len(sample_size_list), 12))
    
    
    for distribution_type in distribution_type_list:
        for k in tqdm(range(n_experiments)):
            for j, max_rounds in enumerate(sample_size_list):
            
                proportion = alpha_proportion(max_rounds)

                mp = UnivariateMaurerPontil(alpha=alpha)

                ebUB = EBUB(max_rounds=max_rounds, alpha=alpha, c1=c1, c2=c2, c3=c3, c4=c4, CS=CS)
                ebLB = EBLB(max_rounds=max_rounds, alpha=proportion*alpha,tilde_alpha=alpha*(1-proportion),
                                  c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, tilde_CS=tilde_CS, CS=CS)

                ebLBupsilon = UnivariateEmpiricalBernsteinBennettLowerBound(max_rounds=max_rounds, 
                                                                            alpha=proportion*alpha,tilde_alpha=alpha*(1-proportion),
                                                                            c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, tilde_CS=tilde_CS, CS=CS)

                decoupled = UnivariateDecoupledEmpiricalBernstein(max_rounds=max_rounds, alpha=proportion*alpha, c1=c1, c2=c2, CS=CS, 
                                                                  tilde_alpha=alpha*(1-proportion), tilde_c1=c1, tilde_c2=c2, tilde_CS=tilde_CS)    


    
                for i in range(max_rounds):
                    if distribution_type == "beta":
                        a = 2
                        b = 6
                        X = np.random.beta(a, b)
                        # if i == 0:
                        #     print(f"Std of beta({a}, {b}): ", np.sqrt(a*b/((a+b)**2*(a+b+1))) )
                    elif distribution_type == "beta1":
                        a = 5
                        b = 5
                        X = np.random.beta(a, b)
                        # if i == 0:
                        #     print(f"Std of beta({a}, {b}): ", np.sqrt(a*b/((a+b)**2*(a+b+1))) )
                    elif distribution_type == "uniform":
                        X = np.random.uniform(0, 1)
                        # if i == 0:
                        #     print(f"Std of uniform: ", 1/np.sqrt(12))
                    else:    
                        raise ValueError("Distribution not implemented")
                    mp(X)
                    decoupled(X)
                    ebUB(X)
                    ebLB(X)
                    ebLBupsilon(X)
            


            
                # print(f"Number of upsilon in bennett: {bennettLB.upsilon_count}")
                # print(f"Number of upsilon in empirical bernstein: {ebLB.upsilon_count}")

                T_reshaped[k, j, 0] = mp.get_center_plus_radius()
                T_reshaped[k, j, 1] = mp.get_center()
                T_reshaped[k, j, 2] = mp.get_center_minus_radius()

                T_reshaped[k, j, 3] = ebUB.get_center_plus_radius()
                T_reshaped[k, j, 4] = ebUB.get_center()

                T_reshaped[k, j, 5] = ebLB.get_center_minus_radius()
                T_reshaped[k, j, 6] = ebLB.get_center()

                T_reshaped[k, j, 7] = ebLBupsilon.get_center_minus_radius()
                T_reshaped[k, j, 8] = ebLBupsilon.get_center()

                T_reshaped[k, j, 9] = decoupled.get_upper_bound()
                T_reshaped[k, j, 10] = decoupled.get_center() 
                T_reshaped[k, j, 11] = decoupled.get_lower_bound()


                           
        with open("./data/" + distribution_type + ".npy", 'wb') as f:
            np.save(f, T_reshaped)

    