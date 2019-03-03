
# coding: utf-8

'''
Copyright Daniel Scott 2019

Background:
MBA students will often be giving excercises to prepare them for dynamic demands
in moden strategic corporate planning. Some well respected simulators - such as
Globestrat - are strong at providing students with uncertainty, but offer little
to no explanation of how metrics are ordered in bottom influence.

Intent:
The script acts as a framework to extract the rank order of available metrics,
and then apply a profit-maximing function to help guide strategy for the user.
The output is a list of recommended decisions to maximize profit while minimizing expense.

Method:
We apply the Nonlinear conjugate gradient method to extract coefficients which
represent the least error from each term, and then Broyden–Fletcher–Goldfarb–Shanno
algorithm to minimize each cost term from the objective function, and output
the optimized solutions.

Disclaimer:
This is meant for educational purposes only and isn't intended to cause
harm to business simulators. Users use it at their own risk. By using this
script, you are agreeing to claim all responsibility for mis-use.

'''

import numpy as np
from scipy.optimize import minimize
import pandas as pd

class Model:
    '''
    Fit the model to the latest available dataset
    '''

    def predict(self,X):

        X_scaled = self.scaler.transform(X)
        result = self.model.predict(X_scaled)
        return result


    def fit(self):

        import seaborn as sns; sns.set()
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        from sklearn import linear_model
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import Normalizer
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import RFE
        import math

        # Select Dataset Here
        df = pd.read_csv('../data/example-dataset.csv')

        df.drop(columns=['att_storage','att_comm','att_image','Market Share','csr budget','cumulative growth volume','Small Operations','wages'],inplace=True)


        # Remove  certain samples from df
        #df = df[df['csr budget']<30]
        #df = df[:-1]
        #df = df.loc[(df.index!=22)]
        #df = df.loc[(df.index!=17)]
        df = df.loc[(df.index!=30)]
        #df = df[df['isCompeting']==1]

        df['margin_no_costs']= df['Company Tablet Price'] * df['volume']

        #df.drop(df.index[20], inplace=True)
        df_not_weird = df[df['isCompeting']==1]
        df_not_weird.drop(columns=['training'], inplace=True)

    #     for column in df_not_weird.columns:
    #         plt.figure(figsize=(10, 10))
    #         sns.scatterplot(x=column,y='margin_no_costs',color='red',data=df_not_weird)
    #         plt.show()

        sns.lineplot(x='Company Tablet Price', y='margin_no_costs', color="green", data=df_not_weird)
        plt.show()

        df.drop(columns=['margin_no_costs'], inplace=True)
        df.drop(columns=['training'], inplace=True)

        y = df.values[:,1]
        X = df.values[:,2:]

        print(list(df))

        # Add scaler for visualize coefs
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X)
        reg = linear_model.Lasso(alpha=0.05).fit(X_scaled, y)

        clf = linear_model.Lasso(alpha=0.05)

        clf.fit(X_scaled,y)


        print(reg.score(X_scaled, y))

        rfe = RFE(estimator=clf, n_features_to_select=1, step=1)
        rfe.fit(X_scaled, y)
        ranking = rfe.ranking_


        df_coefs= pd.DataFrame()
        df_coefs['feature'] = df.columns[2:]
        df_coefs['coef'] = [clf.coef_[i] for i in range(len(clf.coef_))]
        df_coefs['ranking'] = ranking

        print(df_coefs.sort_values(by='ranking', ascending=True))
        plt.figure(figsize=(25, 10))
        sns.scatterplot(x='feature',y='coef',color='red',data=df_coefs)
        plt.show()

        y_pred = reg.predict(X_scaled)

        df_results= pd.DataFrame()
        df_results['y'] = y
        df_results['y_pred'] = y_pred
        df_results['count'] = np.array([x for x in range(len(X))])

        plt.figure(figsize=(10, 10))
        sns.scatterplot(x='count',y='y',color='green',data=df_results)
        sns.scatterplot(x='count',y='y_pred',color='red',data=df_results)

        plt.show()

        self.model = reg
        self.scaler = scaler

        return self




class Optimizer:
    '''
    Used to store boundaries, and sensitivities entered from the use.
    '''

    def __init__(self):

        # Bounds - Held Constant

        self.agency_base = 6
        self.agency_max = 8
        self.agency_marginal_cost = -500

        self.qualIndex_base = 100
        self.qualIndex_max = 180
        self.qualIndex_marginal_cost = -300

        self.ecolIndex_base = 100
        self.ecolIndex_max = 180
        self.ecolIndex_marginal_cost = -300

        self.price_base = 1
        self.price_max = 1700
        self.price_marginal_cost = 1

        self.ad_base = 1
        self.ad_max = 3100
        self.ad_marginal_cost = -1

        self.iftech1_base = 0
        self.iftech1_max = 1
        self.iftech1_marginal_cost = -1500

        self.iftech2_base = 0
        self.iftech2_max = 1
        self.iftech2_marginal_cost = -1500

        self.iftech3_base = 0
        self.iftech3_max = 1
        self.iftech3_marginal_cost = -1500

        self.trainBudget_base = 0
        self.trainBudget_max = 1500
        self.trainBudget_marginal_cost = -1

        self.csrBudget_base = 0
        self.csrBudget_max = 5
        self.csrBudget_marginal_cost = -350



        # Set bounds for each term
        self.bounds = (    (self.agency_base,self.agency_max),
                           (self.qualIndex_base,self.qualIndex_max),
                           (self.ecolIndex_base,self.ecolIndex_max),
                           (self.price_base,self.price_max),
                           (self.ad_base,self.ad_max),
                           (self.iftech1_base,self.iftech1_max),
                           (self.iftech2_base,self.iftech1_max),
                           (self.iftech3_base,self.iftech1_max),
                           (self.trainBudget_base,self.trainBudget_max),
                           (self.csrBudget_base,self.csrBudget_max)
                      )



        self.con1 = {'type': 'ineq','fun': budget_constraint }
        self.cons = [self.con1]


        # Estimate constants

        # In a future version, constants will use a linear regression estimator.
        self.baseTeamSize = 150
        self.baseAverageWage = 65
        self.baseAgencyTeamSize = 5




    def get_surveys(self,sector):
        '''
        Sector may be 'SME, Consumers, or Enterprise'
        Obtained by user input for the given year.
        '''

        self.cumul_growth_volume =30000
        self.sens_price = 180
        self.sens_qual =100
        self.sens_ecol = 100
        self.sens_service = 120
        self.sens_reputation = 120
        self.sens_agencies = 100
        self.attrTech1 =100
        self.attrTech2 =100
        self.attrTech3 =100
        self.competitor_expected_true = 1
        self.competitor_expected_price = 1000
        self.market_share = 38 #expects percentage, not fraction
        self.small_ops = 34
        self.budget = 40000
        return self



    def init_solver_coefs(self):
        '''
        Initialize the solver values

        x[0] Agencies
        x[1] Quality index
        x[2] Ecology index
        x[3] Sector price
        x[4] Geographic-zone advertising
        x[5] ifTech1
        x[6] ifTech2
        x[7] ifTech3
        x[8] Training Budget
        x[9] CSR Budget
        '''



        self.coefs = np.array([    6,
                                   120,
                                   101,
                                   800,
                                   1700,
                                   0,
                                   1,
                                   1,
                                   110,
                                   2
                                  ])


        return self

def budget_constraint(x):
    '''
    This constraint tracks the budget and makes sure decisions do not exceed the requested limit.
    '''


    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    x4 = x[4]
    x5 = x[5]
    x6 = x[6]
    x7 = x[7]
    x8 = x[8]
    x9 = x[9]

    '''
    x[0] Agencies
    x[1] Quality index
    x[2] Ecology index
    x[3] Sector price
    x[4] Geographic-zone advertising
    x[5] ifTech1
    x[6] ifTech2
    x[7] ifTech3
    x[8] Training Budget
    x[9] CSR Budget
    '''

    # Team Size Estimation
    team_size = optimizer.baseTeamSize + (x0 * optimizer.baseAgencyTeamSize)
    total_wages = optimizer.baseAverageWage * team_size


    # Start Expenses --------------------------------
    totalExpense = 0

    # Agency Related Expense (Team Growth)
    totalExpense += (x0 * optimizer.agency_marginal_cost)

    totalExpense += (x1 * optimizer.qualIndex_marginal_cost)
    totalExpense += (x2 * optimizer.ecolIndex_marginal_cost)

    # Advertising Budget
    totalExpense += (x4)

    # Technology Investment
    totalExpense += (x5 * optimizer.iftech1_marginal_cost)
    totalExpense += (x6 * optimizer.iftech2_marginal_cost)
    totalExpense += (x7 * optimizer.iftech3_marginal_cost)

    totalExpense += (total_wages *-1)



    # Training Costs
    totalExpense += x8

    # CSR Costs
    totalExpense += x9

    totalExpense= abs(totalExpense) / 3

    # End Expenses --------------------------------


    # syntax for less-than inequality
    return  totalExpense - optimizer.budget


def objective(x):
    '''
    Objective function to minimize cost/profit ratio. Considers revenues/costs
    associated with each decision.
    '''
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    x4 = x[4]
    x5 = x[5]
    x6 = x[6]
    x7 = x[7]
    x8 = x[8]
    x9 = x[9]

    '''
    x[0] Agencies
    x[1] Quality index
    x[2] Ecology index
    x[3] Sector price
    x[4] Geographic-zone advertising
    x[5] ifTech1
    x[6] ifTech2
    x[7] ifTech3
    x[8] Training Budget
    x[9] CSR Budget
    '''

    # Team Size Estimation
    team_size = optimizer.baseTeamSize + (x0 * optimizer.baseAgencyTeamSize)
    total_wages = optimizer.baseAverageWage * team_size



    # Start Create Terms-----------------------------
    sector_price = x3
    price_compete = (optimizer.competitor_expected_price / x3 ) * optimizer.sens_price
    quality = x1 * optimizer.sens_qual
    tech1 = x5 * optimizer.attrTech1
    tech2 = x6 * optimizer.attrTech2
    tech3 = x7 * optimizer.attrTech3
    wages = (total_wages / team_size) * optimizer.sens_service
    agencies = x0 * optimizer.sens_agencies
    advertising = x4 * optimizer.sens_reputation
    market_share = optimizer.market_share
    small_operations = optimizer.small_ops
    training = (x8 / team_size) * optimizer.sens_service
    competition_included = optimizer.competitor_expected_true
    cumulative_growth_volume = optimizer.cumul_growth_volume
    csr_budget = x9

    # End Create Terms--------------------------------


    # Start Expenses --------------------------------
    totalExpense = 0

    # Agency Related Expense (Team Growth)
    totalExpense += (x0 * optimizer.agency_marginal_cost)

    totalExpense += (x1 * optimizer.qualIndex_marginal_cost)
    totalExpense += (x2 * optimizer.ecolIndex_marginal_cost)


    # Advertising Budget
    totalExpense += (x4)

    # Technology Investment
    totalExpense += (x5 * optimizer.iftech1_marginal_cost)
    totalExpense += (x6 * optimizer.iftech2_marginal_cost)
    totalExpense += (x7 * optimizer.iftech3_marginal_cost)

    totalExpense += (total_wages *-1)

    # Training Costs
    totalExpense += x8

    # CSR Costs
    totalExpense += x9

    # End Expenses --------------------------------



    # Start Revenues ------------------------------
    # Sales Volume Estimator


    '''
    x[0] Agencies
    x[1] Quality index
    x[2] Ecology index
    x[3] Sector price
    x[4] Geographic-zone advertising
    x[5] ifTech1
    x[6] ifTech2
    x[7] ifTech3
    x[8] Training Budget
    x[9] CSR Budget
    '''

    X = np.array([[

        sector_price,
        price_compete,
        quality,
        #tech1,
        #tech2,
        #tech3,
        #wages,
        advertising,
        agencies,
        #market_share,
        #small_operations,
        #training,
        competition_included,
        #cumulative_growth_volume,
        #csr_budget

    ]])




    totalVolume = model.predict(X)


    # Volume * Product Price
    totalRevenue = (totalVolume * x3) / 1000
    # End Revenues ------------------------------

    totalExpense= abs(totalExpense) / 3

    objective_to_minimize = 1 / (totalRevenue-(totalExpense))

    print("Revenue:",totalRevenue,"Expense",totalExpense,'wages',total_wages, 'R-E',(totalRevenue-totalExpense))

    return objective_to_minimize


optimizer = Optimizer()
optimizer.get_surveys('sme')
optimizer.init_solver_coefs()

model = Model()
model.fit()


x = [optimizer.coefs[i] for i in range(len(optimizer.coefs))]



# objective,initial guesses,method, bounds, constraints
sol = minimize(objective,x,method='SLSQP',              bounds=optimizer.bounds,constraints=optimizer.cons)

decisions = np.array(['Agencies',
                    'Quality index',
                    'Ecology index',
                    'Sector price',
                    'Geographic-zone advertising',
                    'ifTech1',
                    'ifTech2',
                    'ifTech3',
                    'Training Budget',
                    'CSR Budget'])

df = pd.DataFrame()
df['Decision'] = decisions
df['value'] = sol.x
df['value'] = df['value'].astype('int')
df
