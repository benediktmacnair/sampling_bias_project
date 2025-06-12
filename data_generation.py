import yaml
import numpy as np
import pandas as pd

'''
Parameters:
1. n_feature
2. n_noise
3. bad_ratio
4. mean_diff
5. var_diff
6. cov
7. noise_var
8. mixture
9. mix_mean_diff
10. min_var_diff
'''

class DataGenerator:
    '''
    input : config_file
    '''
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        
        for key, value in self.config.items():
            setattr(self, key, value)


    def param_check(self):
        for key, value in self.config.items():
            setattr(self, key, value)


        if self.k_con < 1:
            raise ValueError("At least one continuous feature is required")

        if self.con_mean_bad_dif != self.k_con:
            raise ValueError("con_mean_bad_dif requires a list of length 1 or k_con")
        elif self.cat_levels != self.k_cat:
            raise ValueError("cat_levels requires a list of length 1 or k_cat")
        elif self.cat_var_sha != self.k_cat:
            raise ValueError("cat_var_share requires a list of length 1 or k_cat")
        elif self.bin_prob != self.k_bin:
            raise ValueError("bin_prob requires a list of length 1 or k_bin")
        elif self.bin_bad_ratio != self.k_bin:
            raise ValueError("bin_bad_ratio requires a list of length 1 or k_bin")
        elif self.bin_mean_con_dif != self.k_bin:
            raise ValueError("bin_mean_con_dif requires a list of length 1 or k_bin")
        elif self.bin_mean_bad_dif != self.k_bin:
            raise ValueError("bin_mean_bad_dif requires a list of length 1 or k_bin")
        elif self.mixture is True and (self.mix_mean_dif is None or self.mix_var_dif is None):
            raise ValueError("mix_mean_dif and mix_var_dif have to be set if mixture is True")
        
        if self.verbose == True:
            k = self.k_con + self.k_cat + self.k_bin
            print('Simulating ({} x {}) data set'.format(self.n, k))


    def generate_pos_def_matrix(self):
        # Generate symmetric random matrix
        A = np.random.uniform(0, 1, (self.k_con, self.k_con))
        Sigma = np.dot(A, A.T)
        # to make it positive definite
        Sigma += np.eye(self.k_con) * 1e-6

        return Sigma


    def generate_cov(self, combo_vals):
        # Diagonal covariance matrices (IID case)
        if self.iid and self.replicate is None and self.covars is not None:
            sigma1, sigma2 = np.eye(self.k_con), np.eye(self.k_con)
            
            if self.mixture:
                sigma1x, sigma2x = np.eye(self.k_con), np.eye(self.k_con)

        
        if self.iid is False and self.replicate is None and self.covars is None:
            sigma1, sigma2 = self.generate_pos_def_matrix(), self.generate_pos_def_matrix()

            if np.sum(combo_vals) > 0:
                combo_var_dif = np.mean(np.array(self.bin_var_bad_dif) * combo_vals)
            else:
                combo_var_dif = self.con_var_bad_dif

            # Randomly decide which elements to replace
            num_off_diagonal = int((self.k_con**2 - self.k_con) / 2)
            sigma_index = np.random.binomial(1, combo_var_dif, num_off_diagonal).astype(bool)

            # Indices for upper/lower triangle
            upper_indices = np.triu_indices(self.k_con, k=1)
            lower_indices = np.tril_indices(self.k_con, k=-1)

            # Replace upper triangle
            sigma2[upper_indices[0][~sigma_index], upper_indices[1][~sigma_index]] = sigma1[upper_indices[0][~sigma_index], upper_indices[1][~sigma_index]]
            # Replace lower triangle
            sigma2[lower_indices[0][~sigma_index], lower_indices[1][~sigma_index]] = sigma1[lower_indices[0][~sigma_index], lower_indices[1][~sigma_index]]

            # Replace diagonal
            sigma_index_diag = np.random.binomial(1, combo_var_dif, self.k_con).astype(bool)
            sigma2[np.arange(self.k_con)[~sigma_index_diag], np.arange(self.k_con)[~sigma_index_diag]] = sigma1[np.arange(self.k_con)[~sigma_index_diag], np.arange(self.k_con)[~sigma_index_diag]]

            # Ensure positive semi-definite
            ev = np.linalg.eigvalsh(sigma2)
            while not np.all(ev >= -np.sqrt(np.finfo(float).eps) * np.abs(ev[0])):
                sigma2 = self.generate_pos_def_matrix()
                sigma_index = np.random.binomial(1, combo_var_dif, num_off_diagonal).astype(bool)
                
                sigma2[upper_indices[0][~sigma_index], upper_indices[1][~sigma_index]] = sigma1[upper_indices[0][~sigma_index], upper_indices[1][~sigma_index]]
                sigma2[lower_indices[0][~sigma_index], lower_indices[1][~sigma_index]] = sigma1[lower_indices[0][~sigma_index], lower_indices[1][~sigma_index]]
                sigma_index_diag = np.random.binomial(1, combo_var_dif, self.k_con).astype(bool)
                sigma2[np.arange(self.k_con)[~sigma_index_diag], np.arange(self.k_con)[~sigma_index_diag]] = sigma1[np.arange(self.k_con)[~sigma_index_diag], np.arange(self.k_con)[~sigma_index_diag]]
                ev = np.linalg.eigvalsh(sigma2)

            return sigma1, sigma2

        return None, None


    def generate(self):
        dummy = []
        # self.param_check()

        # set arguments if replicating old DGP
        if self.replicate != None:
            n_new = self.n
            seed_new = self.seed

            for key, value in self.config.items():
                setattr(self, key, value)

            self.n = n_new
            self.seed = seed_new

        # Continuos feature has to be at least one
        if self.k_con < 1:
            raise ValueError("At least one continuous feature is required")
            

        if self.verbose == True and self.k_bin > 0:
            print('Generating {} binary features'.format(self.k_bin))

        '''
        Binary Features
        '''

        # Generating combinations based on k_bin
        if self.k_bin == 0:
            dummy = pd.DataFrame({'B1': np.ones(self.n, dtype=int)})
            combos = ['1']
        elif self.k_bin > 0:
            # generate vector of probs
            bin_prob = [self.bin_prob]*self.k_bin
            
            # generate data
            for i in range(self.k_bin):
                # print(i)
                dummy.append(np.random.binomial(n = 1, size = self.n, p = bin_prob[i]))    
            dummy = np.column_stack(dummy)

            # noise injection for binary features
            for i in range(dummy.shape[1]):
                noise = np.random.binomial(1, self.bin_noise_var, self.n)
                dummy[:, i] = np.abs(dummy[:, i] - noise)

            combos = np.unique([''.join(map(str, row)) for row in dummy])
            dummy = pd.DataFrame(dummy, columns=[f'B{i+1}' for i in range(self.k_bin)])
        else:
            raise ValueError("k_bin must be at least 0")
        

        '''
        Continuous Features
        '''

        if self.verbose == True and self.k_con > 0:
            print('Generating {} continuous features'.format(self.k_con))

        # con_params = {
        #     "combo": [],
        #     "means": [],
        #     "covar": [],
        #     "nonlinearities": None
        # }

        # vectors with mean differences
        if self.bin_mean_con_dif is not list:
            self.bin_mean_con_dif = np.repeat(self.bin_mean_con_dif, self.k_bin)
        else:
            self.bin_mean_con_dif = np.array(self.bin_mean_con_dif)

        if self.bin_mean_bad_dif is not list:
            self.bin_mean_bad_dif = np.repeat(self.bin_mean_bad_dif, self.k_bin)
        else:
            self.bin_mean_bad_dif = np.array(self.bin_mean_bad_dif)

        if self.con_mean_bad_dif is not list:
            self.con_mean_bad_dif = np.repeat(self.con_mean_bad_dif, self.k_con)
        else:
            self.con_mean_bad_dif = np.array(self.con_mean_bad_dif)

        if self.bin_bad_ratio is not list:
            self.bin_bad_ratio = np.repeat(self.bin_bad_ratio, self.k_bin)
        else:
            self.bin_bad_ratio = np.array(self.bin_bad_ratio)

        # Continuous feature + Binary feature combinations
        for combo in combos:
            combo_vals = np.array([int(x) for x in combo])

            combo_idx = np.all(dummy == combo_vals, axis=1)
            combo_count = np.sum(combo_idx)

            # Bad ratio assign
            if np.sum(combo_vals) > 0 and self.k_bin != 0:
                combo_bad_ratio = np.mean(np.array(self.bin_bad_ratio) * combo_vals)
            else:
                combo_bad_ratio = self.bad_ratio
         
            # Compute sample sizes
            combo_n1 = int(round(combo_count * combo_bad_ratio))
            combo_n2 = int(round(combo_count * (1 - combo_bad_ratio)))
            if combo_n1 + combo_n2 != combo_count:
                cid = np.random.binomial(1, 0.5)
                if cid == 0:
                    combo_n1 = combo_count - combo_n2
                else:
                    combo_n2 = combo_count - combo_n1

            # Generate specified or load saved means
            if self.replicate is None:
                mu1 = np.repeat(np.sum(combo_vals * self.bin_mean_con_dif), self.k_con)
                mu2 = mu1 + self.con_mean_bad_dif + np.sum(combo_vals * self.bin_mean_bad_dif)

                if self.mixture:
                    mu1x = mu1 + self.mix_mean_dif
                    mu2x = mu2 + self.mix_mean_dif
            else:
                idx = list(combos).index(combo)  # Find the index of the current combo
                mu1 = self.replicate['params']['continuous']['means'][idx][0]
                mu2 = self.replicate['params']['continuous']['means'][idx][1]
                if self.mixture:
                    mu1x = self.replicate['params']['continuous']['means_x'][idx][0]
                    mu2x = self.replicate['params']['continuous']['means_x'][idx][1]
            
            # Generate covariance matrices
            sigma1, sigma2 = self.generate_cov(combo_vals)

            if self.covars is not None:
                sigma1 = self.covars[0]
                sigma2 = self.covars[1]

            if self.mixture:
                sigma1x = sigma1 + self.mix_var_dif
                sigma2x = sigma2 + self.mix_var_dif
            
            
            rng = np.random.default_rng(self.seed)
            X1 = None
            X2 = None

            if combo_n1 > 0:
                X1 = pd.DataFrame(
                    rng.multivariate_normal(mean=mu1, cov=sigma1, size=combo_n1))
                # if self.mixture:
                #     X1x = pd.DataFrame(
                #         rng.multivariate_normal(mean=mu1x, cov=sigma1x, size=round(combo_n1/2))
                #     )
                #     X1 = pd.concat([X1, X1x], ignore_index=True)

            if combo_n2 > 0:
                X2 = pd.DataFrame(
                    rng.multivariate_normal(mean=mu2, cov=sigma2, size=combo_n2)
                )
                # if self.mixture:
                #     X2x = pd.DataFrame(
                #         np.random.multivariate_normal(mean=mu2x, cov=sigma2x, size=round(combo_n2/2))
                #     )
                #     X2 = pd.concat([X2, X2x], ignore_index=True)

            # Generate classes
            if X1 is not None:
                X1['BAD'] = 'BAD'
            if X2 is not None:
                X2['BAD'] = 'GOOD'

            # Merge data frames
            X = pd.concat([X1, X2], ignore_index=True)
            X['BAD'] = X['BAD'].astype('category')
            print(X)
            # df = X.copy()
            # df.columns = [f'V{i+1}' for i in range(self.k_con)] + ['BAD']

            combo_index = np.where(combo_idx)[0]
            if list(combos).index(combo) == 0:
                combined_df = pd.concat([X, dummy.loc[combo_index].reset_index(drop=True)], axis=1)
            else:
                X['B1'] = combo_vals[0]
                X['B2'] = combo_vals[1]
                combined_df = pd.concat([combined_df, X], ignore_index=True)


        # Inject nonlinearities
        if self.con_nonlinear > 0 and round(self.con_nonlinear * self.k_con) > 0:
            nonlinear_vars = np.random.choice(self.k_con, size=round(self.con_nonlinear * self.k_con), replace=False)
            nonlinear_vars = {col: f'NL{col}' for col in nonlinear_vars}
            nonlinear_type = np.random.binomial(1, 0.5, size=len(nonlinear_vars)) + 1
            nonlinear_desc = ['square' if t == 1 else 'cube' for t in nonlinear_type]
            nonlinearities = ', '.join([f"{var}: {desc}" for var, desc in zip(nonlinear_vars, nonlinear_desc)])
            print(nonlinearities)
            # Transform the features
            for i, var in enumerate(nonlinear_vars):
                if nonlinear_type[i] == 1:
                    combined_df[var] = combined_df[var] ** 2  # Square transformation
                    combined_df = combined_df.rename(columns={var: 'squared_{}'.format(var)})
                elif nonlinear_type[i] == 2:
                    combined_df[var] = combined_df[var] ** 3  # Cube transformation
                    combined_df = combined_df.rename(columns={var: 'cubed_{}'.format(var)})
        else:
            nonlinearities = None

        #Inject noise
        for i in range(self.k_con):
            combined_df.iloc[:, i] = combined_df.iloc[:, i] + np.random.normal(0, self.con_noise_var, size=self.n)

        combined_df.to_csv('mockup_data.csv', index=False)
        print(combined_df)

        '''
        Categorical Features and Noise Variables
        '''


data = DataGenerator('config.yaml')
data.generate()
