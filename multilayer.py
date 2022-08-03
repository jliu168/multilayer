# Multilayer calculation tools
# Jack Liu 2022
#   jyliu2@andrew.cmu.edu
#   jli168.7643@gmail.com
# with assistance from David Heson and Dr. William Robertson
# version 2022.7.21

import numpy as np
import matplotlib.pyplot as plt
import more_itertools as mit
import itertools as it
from collections import namedtuple

Bilayer_Recipe = namedtuple('Bilayer_Recipe',
    ['n1', 'n2', 'n_top', 'n_bot', 'h1', 'h2', 'h_defect', 'num_bilayers'])
"""
Container object for the defining values for a bilayer film.

Attributes
----------
n1, h1 : refractive index and depth of first (topmost) component in bilayer
n2, h2 : refractive index and depth of second component in bilayer
n_top, n_bot : refractive index of top / bottommost semi-infinite mediums
h_defect : depth of defect layer
num_bilayers : number of bilayers
"""

R_Data = namedtuple('R_Data', ['theta0', 'lamb', 'R'])
"""Container to specify the reflectivity / transmissivity of a multilayer""" \
"""given values for angle and wavelength of the incident wave."""

##############################################
# REFLECTIVITY / TRANSMISSIVITY CALCULATION
##############################################

def multilayer_RT(theta0, lamb, n, h, dtype='complex'):
    """
    Calculate reflectivity and transmissivity of a multilayer.
    
    Parameters
    ----------
    theta0 : scalar or 1-D array
        Incident angle of incident wave in radians.
    lamb : scalar or 1-D array
        Wavelength of incident wave.
    n : array
        List of refractive indexes of each medium in/around multilayer.
    h : array
        List of heights of each layer in multilayer. (So requires
        that len(h)+2 == len(n).)
    dtype : optional
        Specifies the numpy array dtype used throughout calculations.
    
    Returns
    -------
    R : (2, ...) array
        Array of reflectivity values. Along axis 0 of R, indexes 0 and 1
        contain the values for p- and s-polarization respectively (so that
        you may use tuple assignment like so in your code: (Rp, Rs) = R).
    T : (2, ...) array
        Array of transmissivity values, shaped a la R.
        
    You may pass in inputs of varying type. The shape of the output R and T
    arrays will vary accordingly as follows:
        - [theta0], [lamb] ---> [outputs shape]
        - array, scalar ---> (2, len(theta0))
        - scalar, array ---> (2, len(lamb))
        - both arrays ---> (2, len(lamb), len(theta0)).
        Note here axis -1 corresponds to angle and axis -2 to wavelength.
        - both scalars ---> (2,)
    """
    
    if not isinstance(theta0, (np.ndarray, list)):
        theta0 = np.array([theta0])
    if not isinstance(lamb, (np.ndarray, list)):
        lamb = np.array([lamb]) 
    theta0 = theta0.astype(dtype)
    lamb = np.expand_dims(lamb.astype(dtype), axis=-1)
    n = np.expand_dims(n.astype(dtype), axis=(-1,-2) if n.ndim<2 else -1)
    h = np.expand_dims(h.astype(dtype), axis=(-1,-2))

    sin_theta, cos_theta = calculate_trig(theta0, n)
    r, t = calculate_rt(cos_theta, n)
    delta = calculate_delta(cos_theta, lamb, n, h)
    a, c = calculate_ac(r, delta)

    R = calculate_R(a, c)
    T = calculate_T(cos_theta, n, a, t)
    return np.squeeze(R), np.squeeze(T)

def find_modes(theta0, lamb, R, R_threshold=0.75, width_threshold=None):
    """
    Find modes from reflectivity data.
    """
    
    theta0_is_list = isinstance(theta0, (np.ndarray, list))
    lamb_is_list = isinstance(lamb, (np.ndarray, list))
    if not (theta0_is_list != lamb_is_list):
        raise TypeError("Must have scalar and list among theta0 and lamb")
        
    continuum, scalar = ((theta0, lamb)
                         if theta0_is_list else
                         (lamb, theta0))
    if width_threshold == None:
        width_threshold = 0.05*abs(continuum[-1] - continuum[0])
    
    (Rp, Rs) = R
    p_mode_minimums, s_mode_minimums = [], []
    for (mode_minimums, Rx) in [(p_mode_minimums, Rp),
                                (s_mode_minimums, Rs)]:
        mode_indexes = np.where(Rx < R_threshold)[0]
        for mode in map(list, mit.consecutive_groups(mode_indexes)):
            i1, i2 = mode[0], mode[-1]
            if abs(continuum[i2] - continuum[i1]) < width_threshold:
                i = np.argmin(Rx[mode])
                min_Rx, min_continuum = Rx[mode][i], continuum[mode][i]
                min_theta0, min_lamb = ((min_continuum, lamb) 
                                        if theta0_is_list else
                                        (theta0, min_continuum))
                mode_minimums.append(R_Data(min_theta0, min_lamb, min_Rx))
    return p_mode_minimums, s_mode_minimums

def calculate_R(a, c):
    return np.abs(c[0]/a[0])**2

def calculate_T(cos_theta, n, a, t):
    tp_total, ts_total = np.prod(t, axis=0) / a[0]
    cos_ratio = cos_theta[-1] / cos_theta[0]
    Tp = (cos_ratio * np.conj(n[-1])/np.conj(n[0])).real * np.abs(tp_total)**2
    Ts = (cos_ratio * n[-1]/n[0]).real * np.abs(ts_total)**2
    return np.stack([Tp, Ts], axis=0)

##############################################
# ELECTRIC FIELD PROFILE CALCULATION
##############################################

def multilayer_F(z, theta0, lamb, E0plus, n, h, dtype='complex'):
    """
    Calculate electric field intensity profile of a multilayer.
    
    Parameters
    ----------
    z: scalar or 1-D array
        Depth(s) below first boundary of multilayer at which to
        calculate electric field intensity.
    theta0 : scalar or 1-D array
        Incident angle of incident wave in radians.
    lamb : scalar or 1-D array
        Wavelength of incident wave.
    E0plus: (2,) array
        Amplitude of incident wave. The first and second elements of the
        array are respectively the amplitudes of the p- and s-polarized
        componenents.
    n : array
        List of refractive indexes of each medium in/around multilayer.
    h : array
        List of heights of each layer in multilayer. (So requires
        that len(h)+2 == len(n).)
    dtype : optional
        Specifies the Numpy array dtype used throughout calculations.
    
    Returns
    -------
    F : scalar or array
        Electric field intensity value(s).
    
    You may pass in inputs of varying type. The type and/or shape of the
    output F will vary accordingly as follows:
        - [z], [theta0], [lamb] ---> [output shape]
        - list, scalar, scalar ---> (len(z),)
        (This configuration is likely the most useful in practice.)
        - list, list, scalar ---> (len(z), len(theta0))
        - list, scalar, list ---> (len(z), len(lamb))
        - list, list, list ---> (len(z), len(lamb), len(theta0))
    and if instead z is a scalar in any of the above configurations,
    the output shape will be the same except with axis 0 removed entirely.
    """
    
    if not isinstance(theta0, (np.ndarray, list)):
        theta0 = np.array([theta0])
    if not isinstance(lamb, (np.ndarray, list)):
        lamb = np.array([lamb])
    if not isinstance(z, (list, np.ndarray)):
        z = np.array([z])
    theta0 = theta0.astype(dtype)
    lamb = np.expand_dims(lamb.astype(dtype), axis=-1)
    n = np.expand_dims(n.astype(dtype), axis=(-1,-2) if n.ndim<2 else -1)
    h = np.expand_dims(h.astype(dtype), axis=(-1,-2))
    z = np.expand_dims(z.astype(dtype), axis=(-1,-2))
    E0plus = np.expand_dims(E0plus.astype(dtype), axis=(-1,-2))

    sin_theta, cos_theta = calculate_trig(theta0, n)
    r, t = calculate_rt(cos_theta, n)
    delta = calculate_delta(cos_theta, lamb, n, h)
    a, c = calculate_ac(r, delta)

    Eplus, Eminus = calculate_Eplus_Eminus(E0plus, t, a, c)
    j, delta_z = find_j_delta_z(z, h)

    Eplus_z, Eminus_z = calculate_Eplus_z_Eminus_z(
        cos_theta, lamb, Eplus, Eminus, n, j, delta_z)

    E = calculate_E(cos_theta, sin_theta, Eplus_z, Eminus_z, j)
    F = calculate_F(E0plus, E)
    
    return np.squeeze(F)

def find_j_delta_z(z, h):
    depth = np.empty_like(h, shape=(len(h)+3, *h.shape[1:]))
    depth[0] = np.full(depth.shape[1:], np.NINF)
    depth[1] = np.zeros(depth.shape[1:])
    depth[2:-1] = np.cumsum(h, axis=0)
    depth[-1] = np.full(depth.shape[1:], np.PINF)
    
    j = np.where((depth[:-1, np.newaxis] <= z) & (z < depth[1:, np.newaxis]))[0]
    delta_z = np.expand_dims(np.where(z>=0, z-depth[j], z), axis=1)
    return j, delta_z

def calculate_Eplus_Eminus(E0plus, t, a, c):
    t_cumprod = np.empty_like(t, shape=(len(t)+1, *t.shape[1:]))
    t_cumprod[0] = np.ones(t.shape[1:])
    t_cumprod[1:] = np.cumprod(t, axis=0)
    
    Eplus = t_cumprod * a/a[0] * E0plus
    Eminus = t_cumprod * c/a[0] * E0plus
    return Eplus, Eminus

def calculate_Eplus_z_Eminus_z(cos_theta, lamb, Eplus, Eminus, n, j, delta_z):
    Kz = np.expand_dims(2*np.pi/lamb * n*cos_theta, axis=1)
    Eplus_z = np.swapaxes(Eplus[j]*np.exp(1j*Kz[j]*delta_z), 0, 1)
    Eminus_z = np.swapaxes(Eminus[j]*np.exp(-1j*Kz[j]*delta_z), 0, 1)
    return Eplus_z, Eminus_z

def calculate_E(cos_theta, sin_theta, Eplus_z, Eminus_z, j):
    Eplus_z_p, Eplus_z_s = Eplus_z
    Eminus_z_p, Eminus_z_s = Eminus_z
    Ex = (Eplus_z_p - Eminus_z_p)*cos_theta[j]
    Ey = Eplus_z_s + Eminus_z_s
    Ez = (Eplus_z_p - Eminus_z_p)*sin_theta[j]
    return np.stack([Ex, Ey, Ez], axis=0)

def calculate_F(E0plus, E):
    Ex, Ey, Ez = E
    E0plus_p, E0plus_s = E0plus
    
    Fx = np.abs(Ex)**2 / np.abs(E0plus_p)**2
    Fy = np.abs(Ey)**2 / np.abs(E0plus_s)**2
    Fz = np.abs(Ez)**2 / np.abs(E0plus_p)**2
    for arr in [Fx, Fy, Fz]:
        arr[np.isnan(arr)] = 0.
        
    X = np.abs(E0plus_p)**2 / np.sum(np.abs(E0plus)**2)
    F = X*(Fx+Fz) + (1-X)*Fy
    return F

##############################################
# MATRIX ENTRIES CALCULATION
##############################################

def calculate_trig(theta0, n):
    sin_theta = n[0]/n * np.sin(theta0)
    cos_theta = np.sqrt(1-sin_theta**2)
    return sin_theta, cos_theta

def calculate_rt(cos_theta, n):
    rp = (n[:-1]*cos_theta[1:]-n[1:]*cos_theta[:-1]) / (
        n[:-1]*cos_theta[1:]+n[1:]*cos_theta[:-1])
    rs = (n[:-1]*cos_theta[:-1]-n[1:]*cos_theta[1:]) / (
        n[:-1]*cos_theta[:-1]+n[1:]*cos_theta[1:])
    
    tp = 2*n[:-1]*cos_theta[:-1] / (
        n[:-1]*cos_theta[1:]+n[1:]*cos_theta[:-1])
    ts = 2*n[:-1]*cos_theta[:-1] / (
        n[:-1]*cos_theta[:-1]+n[1:]*cos_theta[1:])
    
    r = np.stack((rp, rs), axis=1)
    t = np.stack((tp, ts), axis=1)
    return r, t

def calculate_delta(cos_theta, lamb, n, h):
    delta = np.expand_dims(2*np.pi/lamb * h*cos_theta[1:-1]*n[1:-1], axis=1)
    return np.concatenate((np.zeros((1, *delta.shape[1:])), delta))

def calculate_ac(r, delta):
    """
    Calculates entries a and c of cumulative products of transfer matrices.
    
    a and c are respectively the top-left and bottom-left entries 
    of the matrix product. a and c are lists, so that indexing a[0]
    gives the entry for the product of all the matrices, a[1] gives the
    entry for the product of all matrices minus the first, etc.
    
    See Ohta and Ishida:
    D[j] = C[j+1] C[j+2] ... C[m+1] = | a[j] b[j] |
                                      | c[j] d[j] |
    """
    
    delta_view = np.lib.stride_tricks.as_strided(delta,
        shape=(delta.shape[0], 2, *delta.shape[2:]),
        strides=(delta.strides[0], 0, *delta.strides[2:])
    )
    C_generator = (np.array([
        [np.exp(-1j*delta_view[i]), r[i]*np.exp(-1j*delta[i])],
        [r[i]*np.exp(1j*delta_view[i]), np.exp(1j*delta_view[i])]
    ]) for i in range(len(delta)-1, -1, -1))
    
    D_i = next(C_generator)
    a = np.empty_like(D_i, shape=(len(delta)+1, *D_i[0,0].shape))
    c = np.empty_like(a)
    a[-1], c[-1] = np.ones(a.shape[1:]), np.zeros(c.shape[1:])
    a[-2], c[-2] = D_i[0,0], D_i[1,0]
    for i in range(3, len(a)+1):
        D_i = np.einsum('ij...,jk...->ik...', next(C_generator), D_i)
        a[-i], c[-i] = D_i[0,0], D_i[1,0]
    return a, c

##############################################
# PLOTTING HELPERS
##############################################

def plot_RT_data(theta0, lamb, R, T, plot=None, lamb_unit=None):
    """
    Helper for plotting output from multilayer_RT calculations.
    
    Parameters
    ----------
    theta0, lamb : incident wavelength and angle input values as passed
        to multilayer_RT. At least one must be an array.
    R, T : reflectivity and transmissivity data as returned by multilayer_RT
    plot : string or list of strings, optional
        Specifices which calculations are plotted. Choose from
        'Rp', 'Rs', 'Tp', 'Ts', or a list/concatenation of any
        of these. Default is to plot all.
    lamb_unit : string, optional 
        Optional label for wavelength units on graph axes, e.g. lamb_unit='nm'.
        
    Returns
    -------
    (fig, ax) or a list of such tuples : matplotlib objects for
        each of the plots generated.
    """
    
    (Rp, Rs), (Tp, Ts) = R, T
    plot_Rp = plot is None or plot=='Rp' or 'Rp' in plot
    plot_Rs = plot is None or plot=='Rs' or 'Rs' in plot
    plot_Tp = plot is None or plot=='Tp' or 'Tp' in plot
    plot_Ts = plot is None or plot=='Ts' or 'Ts' in plot
    theta0_is_list = isinstance(theta0, (np.ndarray, list))
    lamb_is_list = isinstance(lamb, (np.ndarray, list))
    
    plots = []
    if theta0_is_list and lamb_is_list:
        # 3d plotting
        Theta0, Lamb = np.meshgrid(theta0, lamb)
        for (data, cond, title) in zip([Rp, Rs, Tp, Ts],
                                       [plot_Rp, plot_Rs, plot_Tp, plot_Ts],
                                       ['$R_p$', '$R_s$', '$T_p$', '$T_s$']):
            if cond:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                ax.set_title(title)
                ax.set_xlabel(r'$\theta$ [$\degree$]')
                if lamb_unit is None:
                    ax.set_ylabel(r'$\lambda$')
                else:
                    ax.set_ylabel(f'$\\lambda$ [{lamb_unit}]')
                ax.set_zlabel(title)
                ax.plot_surface(Theta0*180/np.pi, Lamb, data, cmap='cividis')
                plots.append((fig, ax))
        return plots if len(plots)>1 else plots[0]
    else:
        # reflectivity / transmission plotting
        if theta0_is_list:
            x_axis = theta0 * 180/np.pi
            x_label = r'$\theta$'
            x_unit = r'[$\degree$]'
        else:
            x_axis = lamb
            x_label = r'$\lambda$'
            x_unit = f'[{lamb_unit}]' if lamb_unit != None else None
        
        if plot_Rp or plot_Rs:
            fig1, ax1 = plt.subplots()
            ax1.set_title(r'$R$ vs. ' + x_label)
            if x_unit is None: ax1.set_xlabel(x_label)
            else: ax1.set_xlabel(x_label + ' ' + x_unit)
            ax1.set_ylabel('$R$')
            for (data, cond, label) in [(Rp, plot_Rp, '$R_p$'), (Rs, plot_Rs, '$R_s$')]:
                if cond: ax1.plot(x_axis, data, label=label)
            ax1.legend()
            plots.append((fig1, ax1))
        if plot_Tp or plot_Ts:
            fig2, ax2 = plt.subplots()
            ax2.set_title(r'$T$ vs. ' + x_label)
            if x_unit is None: ax2.set_xlabel(x_label)
            else: ax2.set_xlabel(x_label + ' ' + x_unit)
            ax2.set_ylabel('$T$')
            for (data, cond, label) in [(Tp, plot_Tp, '$T_p$'), (Ts, plot_Ts, '$T_s$')]:
                if cond: ax2.plot(x_axis, data, label=label)
            ax2.legend()
            plots.append((fig2, ax2))
        return plots if len(plots)>1 else plots[0]

def plot_F_data(z, F, z_unit=None):
    """
    Helper for plotting output from multilayer_F calculations.
    
    Parameters
    ----------
    z : input as passed to multilayer_F. Must be array.
    F : output as return by multilayer_F.
    z_unit : optional string for label units on graph axes.
    
    Returns
    -------
    (fig, ax): matplotlib objects for plot.
    """
    
    fig, ax = plt.subplots()
    ax.set_title('$F(z)$')
    if z_unit is None: ax.set_xlabel('$z$')
    else: ax.set_xlabel(f'$z$ [{z_unit}]')
    ax.set_ylabel('$F(z)$')
    ax.plot(z, F)
    return fig, ax

##############################################
# BILAYER EXPERIMENTS
##############################################

def recipe_to_nh(recipe):
    """Convert a Bilayer_Recipe tuple to n and h arrays for passing """
    """into calculation functions."""

    n1, n2, n_top, n_bot, h1, h2, h_defect, num_bilayers = recipe
    n = np.array([n1 if i%2==1 else n2 for i in range(2*num_bilayers+2)])
    n[0], n[-1] = n_top, n_bot
    h = np.array([h1 if i%2==0 else h2 for i in range(2*num_bilayers)])
    h[-1] = h_defect
    return n, h

def find_recipe_modes(theta0, lamb, all_recipes,
                      R_threshold=0.75, width_threshold=None, lazy=False):
    """
    Catalog the locations of surface modes from a set of bilayer recipes.
    
    Parameters
    ----------
    theta0 : array
        The array 
    lamb : scalar
    all_recipes : iterable of Bilayer_Recipe tuples
        Container for all bilayer recipes to be searched for modes, e.g.
        a list or generator.
    R_threshold: scalar, optional
        The maximum reflectivity that will be considered for modes.
        Default is 0.75.
    width_threshold: scalar, optional
        
        
    Returns
    -------
    recipe_modes : dict
        Dictionary containing the modes found and their associated bilayer
        recipes. Keys are Bilayer_Recipe tuples and values are tuples with
        two lists of R_Data tuples. The first list contains that recipe's
        p-polarization modes, and the second list its s-polarization modes.
    """
    
    if lazy:
        def print_progress(i):
            if i%1000==0:
                print("    On recipe", i+1)
    else:
        all_recipes = list(all_recipes)
        milestone_i = {int(percent*len(all_recipes))
                       for percent in np.arange(0., 1., 0.10)}
        def print_progress(i):
            if i in milestone_i:
                print(f"    On recipe {i+1} / {len(all_recipes)}, "
                      f"{i/len(all_recipes):.0%} done")
        
    recipe_modes = dict()
    print("Looping over all recipes...")
    for i, recipe in enumerate(all_recipes):
        print_progress(i)
        n, h = recipe_to_nh(recipe)
        R, T = multilayer_RT(theta0, lamb, n, h)
        p_modes, s_modes = find_modes(
            theta0, lamb, R, R_threshold, width_threshold)
        if p_modes or s_modes: # as long as either list is nonempty
            recipe_modes[recipe] = (p_modes, s_modes)
    
    print("... Done.")
    return recipe_modes

