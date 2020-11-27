#%%
import numpy as np
import pypsa
import logging
from pypsa.linopt import get_var, define_constraints, linexpr

# %%
def extra_functionality(network, snapshots,g_1):




def daily_scheduler(C,P,alpha,beta,G,g_max=1,g_start=None):
    # C     : CO2 emission intensity
    # P     : Electricity price 
    # alpha : weighting for CO2 emissions
    # beta  : weigthing for Electricity price 
    # G     : Full load hours to schedule in the given day
    # g_max : Maximum power of plant (MW)
    # g_sart: generation in first hour. Should be value between 0 and g_max
    #
    # C and P must contain 24 values 
    # alpha+beta should sum to 1 
    # G must be between 0 and 24 
    #
    # returns:
    # g     : production in the given hours 
    # I     : indexes of optimal hours 
    #
    #
    # This function will minimize the objective
    # g_i*(C_i*alpha + P_i*beta) for all i
    # Subject to the constraint
    # G = sum(g_i) for all i 
    # and further technical constraints 

    # Disable logger 
    logging.disable()

    # Tjek input parameters 
    assert len(C) == 24 and len(P) == 24
    assert alpha+beta == 1
    assert G>0 and G<24*g_max

    # Make C and P correct data format 
    C = np.array(C)
    P = np.array(P)

    # Create the weighted cost using alpha and beta 
    weighted_cost = alpha*C + beta*P

    # Create pypsa model with 24 snapshots 
    model = pypsa.Network()
    model.set_snapshots(range(len(C)))

    # Add a single node 
    model.add('Bus','H2')    


    # Calculate maximum and minimum % load for the given hour
    # This is wehere the starting power is defined as 
    # the p_min and p_max is set to the same value for the first hour
    p_min = np.zeros(24)
    p_max = np.ones(24)
    if g_start is not None :
        p_min[0] = g_start/g_max
        p_max[0] = g_start/g_max

    # Add a generator object with all the technical constraints 
    model.add('Generator','Electrolyzer',
            bus='H2',
            committable=True,
            p_nom=g_max,
            p_nom_extendable=False,
            marginal_cost=weighted_cost,
            p_min_pu = p_min,            # Minimum load when turned on
            p_max_pu = p_max,
            min_up_time = 2,            # Minimum up time in hours 
            ramp_limit_up = 0.3,        # Maximal up ramping speed pr. hour 
            ramp_limit_down = 0.3,      # Maximal down ramping speed pr. hour 
            ramp_limit_start_up = 0.15, # Max ramp up, when starting from 0 
           )

    # Add a load and storage unit, to create a flexible demand, that requires G full load hours 
    # that can be delivered at any hours 
    model.add('Store','store',
            bus='H2',
            e_nom=G)
    load = np.zeros(len(model.snapshots))
    load[-1] = G
    model.add('Load','demand',
                bus='H2',
                p_set=load)

    # Solve model wiht gurobi optimizer
    model.lopf(solver_name="gurobi",pyomo=True)

    # Retrive optimal hours and generation in these hours 
    g = model.generators_t.p['Electrolyzer']
    I =  np.where(g>0)[0]

    return g,I








#%%

if __name__ == '__main__':
    # This section tests the functionality of the daily_scheduler function

    import pandas as pd
    import plotly.graph_objects as go

    # create artificial power price and co2 intensity 
    t_start = '2015-01-01T00:00Z' # start time does not matter here. Total period length is the important factor
    t_end = '2015-01-01T23:00Z'

    hours=pd.date_range(t_start,t_end,freq='H') #'2017-12-31T23:00Z'
    spot_price = [140]
    max_rate = 20
    rate = 0
    for i in range(len(hours)-1):
        rate = rate + (np.random.rand()*2-1)*max_rate
        spot_price.append(spot_price[i-1]+rate)
        
    P = pd.Series(index=hours,data=spot_price)

    CO2_int = [100]
    max_rate = 20
    rate = -10 
    for i in range(len(hours)-1):
        rate = rate + (np.random.rand()*2-1)*max_rate
        new_CO2_int = CO2_int[i-1]+rate
        if new_CO2_int <= 0 :
            new_CO2_int = 0
            rate = 20
        CO2_int.append(new_CO2_int)

    C = pd.Series(index=hours,data=CO2_int)

    alpha = 0.5
    beta = 0.5

    G = 3

    g,I = daily_scheduler(C,P,alpha,beta,G)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours,y=g,name='Optimal production'))
    fig.add_trace(go.Scatter(x=hours,y=C/max(C),name='CO2 intensity'))
    fig.add_trace(go.Scatter(x=hours,y=P/max(P),name='Electricity cost'))
    fig.add_trace(go.Scatter(x=hours,y=(alpha*C + beta*P)/max(alpha*C + beta*P),name='Weighted cost'))

    fig.show()

# %%
