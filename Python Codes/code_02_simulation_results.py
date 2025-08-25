import pandas as pd
import seaborn as sns
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec



class SimulationResults:
    def __init__(self, holdout, curr_accepts, curr_rejects, stat, eval, labeled_rejects = None ):
        self.holdout = holdout.copy()
        self.holdout['Sample'] = 'Population'

        self.curr_accepts = curr_accepts.copy()
        self.curr_accepts['Sample'] = 'Accepts'

        self.curr_rejects = curr_rejects.copy()
        self.curr_rejects['Sample'] = 'Rejects'

        self.stat_list = stat.copy()

        self.eval = eval.copy()
        self.eval['method'] = 'Others'

        self.labeled_rejects = labeled_rejects

        self.min_gen = 5

        self.data = pd.concat([self.holdout, self.curr_accepts, self.curr_rejects], ignore_index=True)


###################################
#                                 
#        DATA DISTRIBUTION
#                                 
###################################


############ Feature Densities

    def _create_density_plot(self, x_col, x_label):
        colors = ['#00BA38', '#FFA500', '#619CFF']

        ax = sns.kdeplot(data=self.data, 
                         x=x_col, 
                         hue='Sample', 
                         fill=True, 
                         alpha=0.3, 
                         palette=colors, 
                         linewidth=1,
                         hue_order=['Accepts', 'Rejects', 'Population'])
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Density')
        ax.set_title('Sampling Bias on {}'.format(x_label), ha='center')

        return ax

    
    def feature_density(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        plt.sca(ax1)
        self._create_density_plot('X1', 'Sampling Bias on X1')

        plt.sca(ax2) 
        self._create_density_plot('X2', 'Sampling Bias on X2')

        plt.tight_layout()
        # plt.savefig(os.path.join(resu_folder, 'sim_feature_densities.pdf'), 
        #         format='pdf', bbox_inches='tight')
        plt.show()



############ Target Density
    def _conditional_density_plot(self, x, y, ax, title, xlabel, ylabel, xlim):

        y_numeric = y.map({"BAD": 1, "GOOD": 0}).astype(float)
        
        # Create bins for x
        if xlim is None:
            xlim = (np.percentile(x, 1), np.percentile(x, 99))
        
        # Filter data within xlim
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x_filtered = x[mask]
        y_filtered = y_numeric[mask]
        
        # Create bins
        n_bins = 50
        bins = np.linspace(xlim[0], xlim[1], n_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate conditional probabilities for each bin
        digitized = np.digitize(x_filtered, bins)
        densities = []
        
        for i in range(1, len(bins)):
            mask_bin = digitized == i
            if mask_bin.sum() > 0:
                density = y_filtered[mask_bin].mean()
            else:
                density = 0
            densities.append(density)
        
        # Plot the conditional density
        ax.plot(bin_centers, densities, linewidth=2, color='blue')
        ax.fill_between(bin_centers, 0, densities, alpha=0.3, color='lightblue')
        
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    def _subplot_feature(self, ax, xlim1):
        self._conditional_density_plot(self.data['X1'], 
                                      self.data['BAD'], 
                                      ax, 
                                      'Target vs X1', 
                                      'X1', 
                                      'Target Density',
                                      xlim1)
        accepts_x1_p01 = np.percentile(self.curr_accepts['X1'], 1)
        ax.axvline(x=accepts_x1_p01, color='red', linestyle='--', linewidth=2)
        return ax


    def target_density(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Feature X1
        xlim1 = (np.percentile(self.data['X1'], 1), np.percentile(self.data['X1'], 99))
        ax1 = self._subplot_feature(ax1, xlim1)

        # Feature X2
        xlim2 = (np.percentile(self.data['X2'], 1), np.percentile(self.data['X2'], 99))
        ax2 = self._subplot_feature(ax2, xlim2)

        plt.tight_layout()
        plt.show()



############ Correlation Plots

# pairs.panels() is a function from a R-Library. Closest alternative in python is as follows

    def pairs_plot(self, which):
        if which == 'accepts':
            data = self.curr_accepts.drop('Sample', axis=1, errors='ignore')
        elif which == 'rejects':
            data = self.curr_rejects.drop('Sample', axis=1, errors='ignore')
        elif which == 'holdouts':
            data = self.holdout.drop('Sample', axis=1, errors='ignore')
        else:
            raise ValueError('choose one for accepts, rejects or holdouts')

        # Create pairplot
        pairplot = sns.pairplot(data[['X1', 'X2', 'BAD']], 
                                diag_kind='hist', 
                                plot_kws={'alpha': 0.3, 's': 20}, 
                                diag_kws={'color': 'grey', 'alpha': 0.7})
        
        pairplot.fig.suptitle('{} - Correlation Matrix'.format(which), y=1.02)
        plt.show()

###################################
#                                 
#        IMPACT ON TRAINING       
#                                 
###################################


############ Decision Bounderies (half baked. To revisit!)

    # def decision_boundaries(self):
        
    #     plt.figure(figsize=(10, 6))
        
    #     # Scatter plot
    #     sns.scatterplot(data=self.data, x='X1', y='X2', hue='sample', 
    #                 palette={'Accepts': '#00BA38', 'Rejects': '#F8766D'},
    #                 s=60, alpha=0.7)
        
    #     plt.xlim(-3, 5)
    #     plt.ylim(-3, 3)
    #     plt.xlabel('x₁', fontsize=12)
    #     plt.ylabel('x₂', fontsize=12, rotation=0, labelpad=15)
    #     plt.title('Decision Boundaries (Simplified)', fontsize=14)
    #     plt.legend(title='Sample')
    #     plt.grid(True, alpha=0.3)
        
    #     plt.tight_layout()
    #     plt.show()


############ Gains from BASL

    def _reshape_data(self, data, column):

        restructured = pd.melt(data,
                    id_vars=['generation'],
                    value_vars=column,
                    var_name='training',
                    value_name='performance')

        restructured = restructured[['generation', 'training', 'performance']]   
        return restructured

    def basl_gain(self):
        stat_basl = self.stat_list
        stat_basl = stat_basl.iloc[self.min_gen:]
        stat_basl['generation'] = stat_basl.index
        loss_due_to_bias = stat_basl['auc_unbiased'] - stat_basl['auc_accepts']
        gain_basl = stat_basl['auc_inference'] - stat_basl['auc_accepts']

        stat_basl['auc_gap'] = loss_due_to_bias
        stat_basl['auc_gap_ri'] = stat_basl['auc_unbiased'] - stat_basl['auc_inference']
        
        # print result
        print('Average loss due to bias: {}'.format(round(st.mean(loss_due_to_bias), 4)))
        print('Performance gains from BASL-based bias correction: {}'.format(round(st.mean(gain_basl), 4)))
        print('Performance gains from BASL-based bias correction: {}'.format((100*round(st.mean(gain_basl/loss_due_to_bias), 4))))
        
        # performance dataset
        perf_basl = self._reshape_data(stat_basl, ['auc_accepts', 'auc_unbiased', 'auc_inference'])     

        # gap dataset
        # gap = self._reshape_data(stat_basl, ['auc_gap', 'auc_gap_ri'])     

        # plot
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(1, 2, width_ratios=[5.5, 4.5], figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])

        colors = ['#00BA38', '#619CFF',  '#FFA500']
        labels = ['Accepts', 'Oracle', 'Accepts + BASL']

        for i, training_type in enumerate(['auc_accepts', 'auc_unbiased', 'auc_inference']):
            subset = perf_basl[perf_basl['training'] == training_type]
            ax1.plot(subset['generation'], subset['performance'], color=colors[i], linewidth=2.5, label=labels[i])

        ax1.set_xlabel('Acceptance Loop Iteration', fontsize=12)
        ax1.set_ylabel('AUC on Holdout Sample', fontsize=12)
        ax1.set_title('(a) Performance', fontsize=14, ha='center', pad=20)
        ax1.grid(True, alpha=0.3)

        legend1 = ax1.legend(title='Scoring Model', 
                            loc='lower right',
                            frameon=True,
                            fancybox=False,
                            edgecolor='black',
                            facecolor='white')
        legend1.get_frame().set_linewidth(1)


        ax2 = fig.add_subplot(gs[0, 1])
        
        ax2.plot(stat_basl['generation'], stat_basl['auc_gap'], color='#00BA38', linewidth=2.5, label='Accepts')
        ax2.plot(stat_basl['generation'], stat_basl['auc_gap_ri'], color='#FFA500', linewidth=2.5, label='Accepts + BASL')
        
        # Add horizontal line at y=0 (Oracle reference)
        ax2.axhline(y=0, color='#619CFF', linestyle='--', linewidth=1, alpha=0.7)
        
        ax2.set_xlabel('Acceptance Loop Iteration', fontsize=12)
        ax2.set_ylabel('AUC Gap', fontsize=12)
        ax2.set_title('(b) Performance Gap', fontsize=14, ha='center', pad=20)
        ax2.grid(True, alpha=0.3)

        legend2 = ax2.legend(title='Scoring Model', 
                            loc='lower right',
                            frameon=True,
                            fancybox=False,
                            edgecolor='black',
                            facecolor='white')
        legend2.get_frame().set_linewidth(1)

        plt.tight_layout()
        plt.show()



###################################
#                                 
#       IMPACT ON EVALUATION     
#                                 
###################################

############ Gains from Bayesian Evaluation

    def _get_best(self, eval, param, text):

        best_holdout_idx = eval['auc_holdout'].idxmax()
        # best_accepts_idx = eval['auc_accepts'].idxmax()
        best_param_idx = eval[param].idxmax()

        eval.at[best_holdout_idx, 'method'] = 'Best on Holdout'
        eval.at[best_param_idx, 'method'] = 'Best on {}'.format(text)

        if best_holdout_idx == best_param_idx:
            duplicate_row = eval.loc[best_holdout_idx].copy()
            duplicate_row['method'] = 'Best on Holdout'
            
            eval = pd.concat([eval, duplicate_row.to_frame().T], ignore_index=True)
            
            eval.at[best_param_idx, 'method'] = 'Best on {}'.format(text)

        return eval
    
    def _subplot_create(self, plot, df, param, title):
        colors = {
        'Best on {}'.format(param): '#00BA38',    
        'Best on Holdout': '#619CFF',  
        'Others': 'gray'              
        }

        for method in df['method'].unique():
            subset = df[df['method'] == method]
            plot.scatter(subset['auc_{}'.format(param)], subset['auc_holdout'], c=colors[method], s=100, marker='o', alpha=0.8, label=method)

        best_holdout_auc = df[df['method'] == 'Best on Holdout']['auc_holdout'].iloc[0]
        best_param_auc = df[df['method'] == 'Best on {}'.format(param)]['auc_holdout'].iloc[0]
        
        plot.axhline(y=best_holdout_auc, color='#619CFF', linestyle='--', linewidth=2, alpha=0.7)
        plot.axhline(y=best_param_auc, color='#00BA38', linestyle='--', linewidth=2, alpha=0.7)

        plot.set_xlabel('AUC on {}'.format(param), fontsize=12)
        plot.set_ylabel('AUC on Holdout Sample', fontsize=12)
        plot.set_title(title, fontsize=14, ha='center', pad=20)
        plot.grid(True, alpha=0.3)

        return plot

    def be_gain(self):
        eval_best_accept = self.eval.copy()
        eval_best_bayes = self.eval.copy()

        eval_best_accept = self._get_best(eval_best_accept, 'auc_accepts', 'accepts')
        eval_best_bayes = self._get_best(eval_best_bayes, 'auc_bayesian', 'bayesian')
        
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(1, 2, width_ratios=[5.5, 4.5], figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1 = self._subplot_create(ax1, eval_best_accept, 'accepts', '(a) Evaluation on Accepts')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2 = self._subplot_create(ax2, eval_best_bayes, 'bayesian', '(b) Evaluation on Bayesian')

        plt.tight_layout()
        plt.show()


    


        
        





