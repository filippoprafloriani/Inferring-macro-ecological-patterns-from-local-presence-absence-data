import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from scipy.stats import norm, gamma
from scipy.optimize import curve_fit
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from sympy import *
import pandas as pd

gen = np.random.default_rng()

def function(n):
    func = binom(n+r-1, n) * g**n * (1-g)**r * (1/(1-(1-g)**r))
    return func

def metropolis(function, param_init, max_steps, burn_in, sigma):
    param_cur = param_init
    func_cur = function(param_cur)
    samples = []
    n_nan = 0

    for n in range(max_steps):
        param_prop = np.random.normal(param_cur,sigma)
        func_prop = function(param_prop)

        if np.isnan(func_prop/func_cur):
            n_nan = n_nan + 1
        elif ((func_prop/func_cur >= 1) or (np.random.uniform(0,1) <= func_prop/func_cur)):
            if param_prop >= 1:
                param_cur = param_prop
                func_cur = func_prop
        else:
            pass
            
        if (n > burn_in):
            samples.append(param_cur)
    
    return samples



def generate_abundances(distribution, metropolis_, load_file, parameters, plot_hist, parameters_metropolis = None):
    if distribution == 'NB':
        r, xi, size_species = parameters
        
        if metropolis_:
            inital_step, max_steps, burn_in, sigma_ = parameters_metropolis
            samples = metropolis(function, inital_step, max_steps, burn_in, sigma_)
            samples_S_nb = gen.choice(samples, size_species)
            
            abundances = np.ceil(np.array(samples_S_nb)).astype(int)

            if np.any(abundances == 0):
                print('Error, some species have 0 populations')

        if load_file:
            abundances = np.loadtxt('abundances_NB.txt').astype(int)
            if np.any(abundances == 0):
                print('Error, some species have 0 populations')
                
        else:
            abundances = gen.negative_binomial(r, 1-xi, size = size_species)  #requires n,p so r,1-g
            
            if np.any(abundances == 0):
                index = np.array(np.where(abundances == 0)).flatten()
                for i in index:
                    prop = gen.negative_binomial(r, 1-xi)
                    while prop < 1 :
                        prop = gen.negative_binomial(r, 1-xi)
                    abundances[i] = prop

            if np.any(abundances == 0):
                print('Error, some species have 0 populations')


    if distribution == 'LN':
        mu, sigma, size_species = parameters
        
        abundances = gen.lognormal(mean = mu, sigma = sigma, size = size_species)
        abundances = np.ceil(np.array(abundances)).astype(int)
        if np.any(abundances == 0):
                print('Error, some species have 0 populations')
            
    #Plot histogram
    if plot_hist:
        counts_hist, bins_hist, __ = plt.hist(abundances, bins = int(np.sqrt(size_species)))
        plt.xlabel('n', fontsize = 15)
        plt.ylabel('Number of species', fontsize = 15)
        plt.title('RSA', fontsize = 15)
        plt.show()
    
    return abundances, counts_hist, bins_hist



def generate_cluster_centers_poisson(intensity, side_matrix_):   
    # Generate the number of points according to a Poisson distribution
    num_points = np.random.poisson(intensity * side_matrix_**2)
    
    # Generate random positions for the points
    x_coords = np.random.randint(0, side_matrix_, num_points)
    y_coords = np.random.randint(0, side_matrix_, num_points)
    
    return np.column_stack((x_coords, y_coords))




def compute_valid_side_cell(side_matrix_, initial_side_cell_):
    side_cell_length = []
    for side_length in range(initial_side_cell_, side_matrix_):
        if side_matrix_ % side_length == 0:
            side_cell_length.append(side_length)

    return side_cell_length



def divide_matrix(matrix, n_cell):
    # Get the dimensions of the original matrix
    n, m = matrix.shape
    
    # Calculate the number of rows and columns in each cell
    C = int(np.sqrt(n_cell))
    rows_per_cell = n // C
    cols_per_cell = m // C
    
    # Reshape the matrix into C cells
    cells = []
    for i in range(0, n, rows_per_cell):
        for j in range(0, m, cols_per_cell):
            cell = matrix[i:i+rows_per_cell, j:j+cols_per_cell]
            cells.append(cell)
    
    return cells



def create_M_matrix_in_silico(abundances_nb, size_species, side_matrix, side_cell, p, random_placement, try_use_more_indiv, plot_matrix):
    #Generate an array where we have as many id number species as population
    indiv_per_species = [[i+1]*abundances_nb[i] for i in range(len(abundances_nb))]
    indiv_per_species = np.array([item for sublist in indiv_per_species for item in sublist])  

    if random_placement:
        #Define the grid
        M = np.zeros(side_matrix*side_matrix)
        M[:len(indiv_per_species)] = indiv_per_species
        gen.shuffle(M)
        M_matrix = M.reshape(side_matrix, side_matrix)

    else:
        M_matrix = np.zeros((side_matrix,side_matrix))

        rho = 6e-5   #intensity poisson cluster 
        cluster_centers = generate_cluster_centers_poisson(rho, side_matrix)
        cluster_prefix = 5000
        M_matrix[cluster_centers[:, 0], cluster_centers[:, 1]] = cluster_prefix
        parent_index = np.random.randint(0, len(cluster_centers), len(indiv_per_species))
        
        sigma_cluster = 15  #for the multivariate gaussian
        
        if try_use_more_indiv:
            factor_ = 100       #for moving gaussian if necessary
            threshold = 5     #threshold for trying valid position
            threshold_exit = 10   #threshold for loosing an individual
            loss = 0   #count of lost individuals
    
            for i in tqdm(range(len(parent_index)), desc="Progress"):  #len(parent_index)=len(indiv_per_species)   #tqdm for progress bar
                idx = parent_index[i]
                loc_idx = tuple(gen.multivariate_normal(cluster_centers[idx], sigma_cluster**2 * np.eye(2), size = 1).flatten().astype(int))
                
                count = 0    
                while (np.any(np.array(loc_idx) >= side_matrix) or np.any(np.array(loc_idx) < 0) or ((M_matrix[loc_idx] > 0) and (M_matrix[loc_idx] < 5000))):  #check if the location of the individual within its cluster is free
                    count += 1
                    if count < threshold:   #try at least threshold times to find a position
                        loc_idx = tuple(gen.multivariate_normal(cluster_centers[idx], sigma_cluster**2 * np.eye(2), size = 1).flatten().astype(int))
                    if ((count >= threshold) and (count < threshold_exit)):    #if not found, move the mean of the multivariate gaussian of a factor and try until threshold_exit times
                        loc_idx = tuple(gen.multivariate_normal(cluster_centers[idx] + factor_, sigma_cluster**2 * np.eye(2), size = 1).flatten().astype(int))
                    if (count >= threshold_exit):   #if no position found at all, accept the position and loose an individual of some species
                        loss += 1
                        while (np.any(np.array(loc_idx) >= side_matrix) or np.any(np.array(loc_idx) < 0)):   #be sure to stay inside the area
                            loc_idx = tuple(gen.multivariate_normal(cluster_centers[idx], sigma_cluster**2 *np.eye(2), size = 1).flatten().astype(int)) 
                        break
        
                M_matrix[loc_idx] = indiv_per_species[i]

        else:
            for i in tqdm(range(len(parent_index)), desc="Progress"):  #len(parent_index)=len(indiv_per_species)   #tqdm for progress bar
                idx = parent_index[i]
                loc_idx = tuple(gen.multivariate_normal(cluster_centers[idx], sigma_cluster**2 * np.eye(2), size = 1).flatten().astype(int))
                
                while (np.any(np.array(loc_idx) >= side_matrix) or np.any(np.array(loc_idx) < 0)):
                    loc_idx = tuple(gen.multivariate_normal(cluster_centers[idx], sigma_cluster**2 * np.eye(2), size = 1).flatten().astype(int))
            
                M_matrix[loc_idx] = indiv_per_species[i]

        
        M_matrix[M_matrix==cluster_prefix] = 0
        M = M_matrix.flatten()

    
    if plot_matrix:
        colors = plt.cm.YlGn(np.linspace(0, 1, 256))  #Use YlGn cmap as base
        colors[0] = [1, 1, 1, 1]                      #Set color for zero values to white
        cmap = ListedColormap(colors)
        plt.imshow(M_matrix, cmap=cmap, interpolation='nearest')
        plt.colorbar()
        plt.show()


    return M, M_matrix



def create_M_matrix_from_database_BCI(file_name, check_only_alive = False, plot_matrix = True):

    df = pd.read_csv(file_name)
    if check_only_alive:
        df = df[df['status'] == 'A']
    
    df.dropna(subset=['gx'], inplace=True)
    df.dropna(subset=['gy'], inplace=True)
    
    entries = df[['gx', 'gy']].values
    entries = entries.astype(int)
    
    df['x'] = entries[:,0]
    df['y'] = entries[:,1]
    
    species_name = np.unique(df['sp'])  #find unique names
    species_index = np.arange(1, np.unique(df['sp']).shape[0]+1)  #assign a number to each name 1,2,3, ..., S_tot
    
    size_species = len(species_index)  #tot number of species S_tot
    
    df_species_index = [species_index[np.where(species_name == i)][0] for i in df['sp']]   #make the correspondence between number and name
    df['idx'] = df_species_index
    
    M_matrix = np.zeros((int(max(df['x']))+1, int(max(df['y']))+1)) 
    M_matrix[df[['x', 'y']].values[:,0], df[['x', 'y']].values[:,1]] = df['idx'].values   #put species in matrix

    if plot_matrix:
        colors = plt.cm.YlGn(np.linspace(0, 1, 256))  #Use YlGn cmap as base
        colors[0] = [1, 1, 1, 1]                      #Set color for zero values to white
        cmap = ListedColormap(colors)
        plt.imshow(M_matrix, cmap=cmap, interpolation='nearest')
        plt.colorbar()
        plt.show()

    return M_matrix, size_species



def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) ** 0.5

def generate_points(num_points, side_matrix, min_distance):
    grid_size = (side_matrix, side_matrix)
    points = []
    while len(points) < num_points:
        x = np.random.randint(0, grid_size[0])
        y = np.random.randint(0, grid_size[1])
        new_point = (x, y)
        if all(distance(new_point, existing_point) >= min_distance for existing_point in points):
            points.append(new_point)
    return points



def create_M_matrix_from_database_BIRDS(file_name, side_matrix, random_placement, plot_matrix = True):
    
    df = pd.read_excel(file_name)
    names = np.unique(df['NAME'])

    cluster_centers = 0
    site_per_species = []
    abundances = np.zeros(len(names))
    for i in range(len(names)):
        abundances[i] = np.sum(df[df['NAME'] == names[i]]['N'])
    
    indiv_per_species = [[i+1]*abundances.astype(int)[i] for i in range(len(abundances))]
    indiv_per_species = np.array([item for sublist in indiv_per_species for item in sublist])

    size_species = len(abundances)
    
    if random_placement:
        M = np.zeros(side_matrix*side_matrix)
        M[:len(indiv_per_species)] = indiv_per_species
        gen.shuffle(M)
        M_matrix = M.reshape(side_matrix, side_matrix)

    else:
        df['idx_site'] = [0 for i in range(len(df))]
    
        site = []
        for i in range(0,100,10):
            site.append([int(i/10)]*len(df[(df['SITE']/10000 > i) & (df['SITE']/10000 < 10+i)]))
        
        site_all = np.array([item for sublist in site for item in sublist])  
        df['idx_site'] = site_all

        for i in range(len(names)):
            for j in range(len(np.unique(df['idx_site']))):
                site_per_species.append([np.unique(df['idx_site'])[j]] * np.sum(df[(df['NAME'] == names[i]) & (df['idx_site'] == j)]['N']))

        site_per_species = np.array([item for sublist in site_per_species for item in sublist])

        min_distance = 300
        num_points = len(np.unique(site_all))
        M_matrix = np.zeros((side_matrix,side_matrix))
        
        cluster_prefix = 5000
        cluster_centers = np.array(generate_points(num_points, side_matrix, min_distance))
        M_matrix[cluster_centers[:, 0], cluster_centers[:, 1]] = cluster_prefix


        for i in tqdm(range(len(site_per_species))):
            idx = site_per_species[i]
            loc_idx = tuple(gen.multivariate_normal(cluster_centers[idx], min_distance**2 * np.eye(2), size = 1).flatten().astype(int))
        
            while (np.any(np.array(loc_idx) >= side_matrix) or np.any(np.array(loc_idx) < 0)):
                loc_idx = tuple(gen.multivariate_normal(cluster_centers[idx], min_distance**2 * np.eye(2), size = 1).flatten().astype(int))
        
            M_matrix[loc_idx] = indiv_per_species[i]

        M_matrix[M_matrix==cluster_prefix] = 0
        
    if plot_matrix:
        colors = plt.cm.YlGn(np.linspace(0, 1, 256))  #Use YlGn cmap as base
        colors[0] = [1, 1, 1, 1]                      #Set color for zero values to white
        cmap = ListedColormap(colors)
        plt.imshow(M_matrix, cmap=cmap, interpolation='nearest')
        plt.colorbar()
        plt.show()

    return M_matrix, abundances, size_species, indiv_per_species, site_per_species, cluster_centers, df



def create_MxS_matrix_and_sampling(M_matrix, n_cell, size_species, p):
    cells = divide_matrix(M_matrix, n_cell)
    M_S_matrix = np.zeros((len(cells), size_species))
    for i in range(len(cells)):
        M_S_matrix[i, np.unique(cells[i], return_counts=True)[0][1:].astype(int)-1] = np.unique(cells[i], return_counts=True)[1][1:] 
        
    num_rows_to_sample = int(p * M_S_matrix.shape[0])
    sampled_rows_indices = np.random.choice(M_S_matrix.shape[0], num_rows_to_sample, replace=False)
    sampled_matrix = M_S_matrix[sampled_rows_indices]
    sampled_matrix_species_info = np.copy(sampled_matrix)    
    
    #Convert into binary matrix
    sampled_matrix[sampled_matrix != 0] = 1

    return M_S_matrix, sampled_matrix_species_info, sampled_matrix



def sac_function(p, r_fit, xi_fit, S_star_):
    func = S_star_ * (1-(1-(p*xi_fit/(1+xi_fit*(p-1))))**r_fit)/(1-(1-xi_fit)**r_fit)
    return func



def compute_fit_SAC(sampled_matrix, r_init_prop, xi_init_prop, plot):
    tot_cell = sampled_matrix.shape[0]
    tot_species = np.count_nonzero(np.sum(sampled_matrix, axis = 0))

    sac = np.zeros(tot_cell)
    sac[0] = np.mean(np.sum(sampled_matrix, axis = 1))   #note that in index 0 i have k = 1 and so on, basically if i see under here this is like summing over axis = 0 (get array of size = size_species) and computing the number of elements != 0 for each row singularly (so sum each row), and then summing all and normalizing to num_cells (get mean)s
    sac[-1] = tot_species                                #already know the total number of species in sample of whole area

    rel_err_array = np.zeros(tot_cell)
    rel_err_array[0] = (sac[0] - tot_species)/tot_species * 100
    
    sub_sample_area = np.linspace(1, tot_cell, tot_cell)/tot_cell
    
    n_rand_sample_comb = 100
    for k in range(1,tot_cell-1):  # p_k = k/M*
        check_species = 0
        for rsc in range(n_rand_sample_comb):
            sample_comb = gen.choice(tot_cell, k+1, replace = False)
            sub_sampled_matrix = sampled_matrix[sample_comb]
            check_species_among_sub_samples = np.sum(sub_sampled_matrix, axis = 0)
            check_species += len(check_species_among_sub_samples[check_species_among_sub_samples != 0])
            
        sac[k] = check_species/n_rand_sample_comb
        rel_err_array[k] = (sac[k] - tot_species)/tot_species * 100

    S_ = sac[-1]  #S* 
    param_fitted, cov_fitted = curve_fit(lambda x, r_fit, xi_fit: sac_function(x, r_fit, xi_fit, S_), sub_sample_area, sac, p0 = [r_init_prop, xi_init_prop])
    errors_ = np.sqrt(np.diag(cov_fitted))

    if plot:
        #Plot
        fig = plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 1)
        plt.plot(sub_sample_area, sac, label = 'Empirical curve')
        plt.legend()
        plt.xlabel('p', fontsize = 15)
        plt.ylabel('Number of species', fontsize = 15)
        plt.title('SAC', fontsize = 15)
        
        p_fit = np.linspace(1, tot_cell, 1000)/tot_cell
        plt.subplot(1, 2, 2)
        plt.scatter(sub_sample_area, sac, label = 'Data')
        plt.plot(p_fit, sac_function(p_fit, *param_fitted, S_), color = 'orange', label = 'Fit')
        plt.legend()
        plt.xlabel('p', fontsize = 15)
        plt.ylabel('Number of species', fontsize = 15)
        plt.title('SAC', fontsize = 15)
        
        plt.tight_layout()
        plt.show()

    return sub_sample_area, sac, S_, param_fitted, cov_fitted, errors_, rel_err_array



def plot_err_array(sub_sample_area, rel_err_array):
    plt.plot(sub_sample_area, rel_err_array, label = 'Relative error')
    plt.legend()
    plt.xlabel('p', fontsize = 15)
    plt.ylabel('Relative error', fontsize = 15)
    plt.title('Relative error during algorithm', fontsize = 15)
    plt.show()



def compute_derivative_S_real():
    xi__d, r__d, p__d, S_star__d = symbols('xi r p S^*')
    init_printing(use_unicode=True)
    
    der_r = diff(S_star__d * (1-(1-(xi__d/(p__d + xi__d*(1-p__d))))**r__d)/(1-(1-xi__d)**r__d), r__d)   #derivative wrt r
    der_xi = diff(S_star__d * (1-(1-(xi__d/(p__d + xi__d*(1-p__d))))**r__d)/(1-(1-xi__d)**r__d), xi__d)  #derivative wrt xi
    
    der_r_ = lambdify([xi__d, r__d , p__d, S_star__d], der_r)
    der_xi_ = lambdify([xi__d, r__d , p__d, S_star__d], der_xi)
    
    return der_r_, der_xi_


def compute_real_parameters(r_, xi_, errors_, p, S_star_, S_original, plot_real):
    #Compute real parameters
    r_real = r_
    xi_real = xi_/(p + xi_*(1-p))
    S_real = S_star_ * (1-(1-xi_real)**r_)/(1-(1-xi_)**r_)

    #Compute error of S_real
    der_r, der_xi = compute_derivative_S_real()
    
    der_r_val = der_r(xi_real,r_real,p,S_star_)
    der_xi_val = der_xi(xi_real,r_real,p,S_star_)
    
    err_S_real = np.sqrt(der_r_val**2 * errors_[0]**2 + der_xi_val**2 * errors_[1]**2)   #err_r_fit = errors[0], err_xi_fit = errors[1]

    #Compute relative percentage error and its error
    rel_err = (S_real - S_original)/S_original * 100
    err_rel_err = err_S_real/S_original * 100

    if plot_real:
        p_list = np.linspace(p, 1, 1000)
        plt.plot(p_list, sac_function(p_list, r_real, xi_real, S_real), label = 'Real curve')
        plt.legend()
        plt.xlabel('p', fontsize = 15)
        plt.ylabel('Number of species', fontsize = 15)
        plt.title('SAC', fontsize = 15)
        plt.show()

    return r_real, xi_real, S_real, err_S_real, rel_err, err_rel_err



def compute_relative_error_n_simulations(silico_data, model, M_S_matrix, size_S, p, n_sim, whole_forest = False):
    r_list = []
    xi_list = []
    S_star_list = []
    S_real_list = []
    errors_list_r = []
    errors_list_xi = []
    
    #rel_err_list = []
                        
    for i in tqdm(range(n_sim)):
        plot_matrix = False
        #M_S_matrix, sampled_matrix_species_info, sampled_matrix = create_MxS_matrix_and_sampling(M_matrix, num_cell, size_S, p)

        if silico_data:
            if model == 'NB':
                r_init, xi_init = 0.2, 0.999
                param_init_ = [r_init, xi_init]
                
            if model == 'LN':
                mu_init, sigma_init = 4, 0.8
                param_init_ = [mu_init, sigma_init]
        else:
            r_init, xi_init = 0.2, 0.999
            param_init_ = [r_init, xi_init]

        if whole_forest:
            M_S_matrix_ = np.copy(M_S_matrix)
            M_S_matrix_[M_S_matrix_ != 0] = 1
            sub_sample_area, sac, S_star, param_fitted, cov_fitted, errors, rel_err_array = compute_fit_SAC(M_S_matrix_, *param_init_, plot = False)

        else:
             #Just sampling in another way
            num_rows_to_sample = int(p * M_S_matrix.shape[0])
            sampled_rows_indices = np.random.choice(M_S_matrix.shape[0], num_rows_to_sample, replace=False)
            sampled_matrix = M_S_matrix[sampled_rows_indices]
            sampled_matrix[sampled_matrix != 0] = 1
            
            sub_sample_area, sac, S_star, param_fitted, cov_fitted, errors, rel_err_array = compute_fit_SAC(sampled_matrix, *param_init_, plot = False)
            
        r_real, xi_real, S_real, err_S_real, rel_err, err_rel_err = compute_real_parameters(param_fitted[0], param_fitted[1], errors, p, S_star, size_S, plot_real = False)
    
        r_list.append(r_real)
        xi_list.append(xi_real)
        S_star_list.append(S_star)
        S_real_list.append(S_real)
        errors_list_r.append(errors[0])
        errors_list_xi.append(errors[1])
        
        #rel_err_list.append(rel_err)
        #err_S_real_list.append(err_S_real)
        #err_rel_err_list.append(err_rel_err)
            
    
    #Propagate errors    
    der_r, der_xi = compute_derivative_S_real()
    
    factor_error = 0
    for i in range(len(r_list)):
        der_r_, der_xi_ = compute_derivative_S_real()

        der_r = der_r_(xi_list[i],r_list[i],p,S_star_list[i])
        der_xi = der_xi_(xi_list[i],r_list[i],p,S_star_list[i])
    
        factor_error += der_r**2 * errors_list_r[i]**2 + der_xi**2 * errors_list_xi[i]**2

    S_real_mean = np.mean(S_real_list)   #evaluate mean S_real
    err_S_real_mean = np.sqrt(factor_error)/len(r_list)   #evaluate error mean S_real
    
    rel_error_mean = (S_real_mean - size_S)/size_S * 100   #evaluate mean relative error
    rel_error_error_mean = err_S_real/size_S * 100        #evaluate error mean relative error

    r_mean = np.mean(r_list)
    xi_mean = np.mean(xi_list)

    err_r_mean = np.sqrt(np.sum(errors_list_r))/len(errors_list_r)
    err_xi_mean = np.sqrt(np.sum(errors_list_xi))/len(errors_list_xi)
    
    return S_real_mean, err_S_real_mean, rel_error_mean, rel_error_error_mean, r_mean, err_r_mean, xi_mean, err_xi_mean    



def check_norm(x, f_x):
    norm_ = np.diff(x)[0]*np.sum(f_x)
    return norm_



def RSA_NB(n, r_param, xi_param):
    func = binom(n+r_param-1, n) * xi_param**n * (1-xi_param)**r_param * (1/(1-(1-xi_param)**r_param))    
    norm_ = check_norm(n, func)
    return func, norm_



def compute_RSA(M_matrix, r_param, xi_param):
    M = M_matrix[M_matrix !=0]
    idx_species, abundances = np.unique(M, return_counts=True)
    
    n_array = np.arange(1,max(abundances))
    RSA_values, norm_ = RSA_NB(n_array, r_param, xi_param)
    RSA_values = RSA_values/norm_
    
    counts, bins_edges = np.histogram(abundances, bins = np.logspace(0,np.log2(max(abundances)) + 0.1,15, base = 2))
    bins_centers = (bins_edges[1:] + bins_edges[:-1])/2
    
    plt.scatter(bins_centers,np.cumsum(counts)/len(abundances), label = 'Data')
    plt.plot(n_array, np.cumsum(RSA_values), label = 'Analytical RSA')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Individuals', fontsize = 15)
    plt.ylabel('Frequency', fontsize = 15)
    plt.title('RSA', fontsize = 15)
    plt.show()



def RSO_distribution(M_matrix, num_cell, size_S, p, n_bins, plot = False):
    M_S_matrix, sampled_matrix_species_info, sampled_matrix = create_MxS_matrix_and_sampling(M_matrix, num_cell, size_S, p)
    M_S_matrix[M_S_matrix != 0] = 1
    val = np.sum(M_S_matrix, axis = 0)

    counts, bins_edges = np.histogram(val, n_bins)
    
    if plot:
        plt.stairs(counts, bins_edges, fill = True)
        plt.show()

    fig = plt.figure(figsize = (12,5))

    plt.subplot(1,2,1)
    bins_center = (bins_edges[1:] + bins_edges[:-1])/2
    plt.scatter(bins_center, np.cumsum(counts)/size_S)
    plt.axhline(1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Occurrences', fontsize = 15)
    plt.ylabel('Frequency', fontsize = 15)
    plt.title('RSO', fontsize = 15)

    plt.subplot(1,2,2)
    plt.scatter(bins_center, counts/size_S)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Occurrences', fontsize = 15)
    plt.ylabel('Frequency', fontsize = 15)
    plt.title('RSO', fontsize = 15)

    plt.tight_layout()
    plt.show()

    return bins_center, counts



def RSO_distribution_count_occupied_plot(M_matrix, num_cell, size_S, p, plot = False):
    M_S_matrix, sampled_matrix_species_info, sampled_matrix = create_MxS_matrix_and_sampling(M_matrix, num_cell, size_S, p)
    M_S_matrix[M_S_matrix != 0] = 1
    val = np.sum(M_S_matrix, axis = 0)
    val = val[val!=0]

    number_occupied_cell, count_number_occupied_cell = np.unique(val, return_counts=True)
    
    fig = plt.figure(figsize = (12,5))

    plt.subplot(1,2,1)
    plt.scatter(number_occupied_cell, np.cumsum(count_number_occupied_cell)/size_S)
    plt.axhline(1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Occurrences', fontsize = 15)
    plt.ylabel('Frequency', fontsize = 15)
    plt.title('RSO', fontsize = 15)

    plt.subplot(1,2,2)
    plt.scatter(number_occupied_cell, count_number_occupied_cell/size_S)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Occurrences', fontsize = 15)
    plt.ylabel('Frequency', fontsize = 15)
    plt.title('RSO', fontsize = 15)

    plt.tight_layout()
    plt.show()

    return number_occupied_cell, count_number_occupied_cell

