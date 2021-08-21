from scipy.integrate import simps
import tensorflow as tf
import numpy as np
import math
from scipy.stats import norm
import scipy.stats
import pandas as pd
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.regularizers import l2
from scipy.stats import norm
import tensorflow as tf
import scipy.stats
import cv2
import imageio
import mdn

def find_closest_element(y: float, arr: np.ndarray):
    '''
    Find index of element closest to y from array arr
    
    Input:
        - y: value to which we want to find the closest value
        - arr: array in which we want to find the closest value
    
    Output:
        - index: index where element is closest to y in arr
    '''
    
    index = np.searchsorted(arr,y)
    if (index >= 1) & (index < arr.shape[0]):
        res = [arr[index - 1], arr[index]]
    elif (index < arr.shape[0]):
        return np.array(index)
    else:
        return np.array(index - 1)

    if res[0] == res[1]:
        return np.array(index - 1)
    else:
        diff_pre = np.abs(y-res[0])
        diff_aft = np.abs(y-res[1])
        if diff_pre == diff_aft:
            return np.array(index - 1)
        else:
            return index - 1 if diff_pre < diff_aft else index
def Fy(y, density):
    '''
    Distribution function at y of numerical cdf
    Input:
        - y: scalar, value at which cdf should be evaluated
        - density: dataframe with columns: axes, cdf
    
    Output: 
        - integral: cdf at y
    '''
    integral = density.loc[find_closest_element(y, density['axes']),'cdf']
    return(integral)  

def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-((x_m.T).dot(np.linalg.inv(covariance)).dot(x_m)) / 2))

class DensityPredictor:
    '''
    Class to predict densities from images
    '''
    
    def __init__(self, method, model, p):
        ''' initialize trained model
        choice of:
        - precise learner: 'precise'
        - imprecise learner: 'imprecise'
        Parameters
        ----------
        p: int
            number of coefficients of last hidden layer
        '''
        self.model = model
        self.method = method
        self.p = p
        
        if self.model == 'precise':
            self.extracted_coefficients_path_beta_dnn = '../../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/beta/'
            self.extracted_coefficients_path_beta_hmc = '../../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/beta/'
            self.extracted_coefficients_path_beta_va = '../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/beta/'
        if self.model == 'imprecise':
            self.extracted_coefficients_path_beta = '../../../data/commaai/extracted_coefficients/20201021_unrestr_gaussian_resampled/beta/'
            self.extracted_coefficients_path_beta_hmc = '../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/beta/'
            self.extracted_coefficients_path_beta_va = '../../../data/commaai/extracted_coefficients/20201027_filtered_gaussian_resampled/beta/'
    
    def load_bzeta_model(self):
        
        print('building model ...')
        Input = tf.keras.layers.Input(shape=(66, 200, 3,), name='image')
        x = Conv2D(24, kernel_size=(5, 5), activation='relu', strides=(2, 2))(Input)
        x = BatchNormalization()(x)
        x = Conv2D(36, kernel_size=(5, 5), activation='relu', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(48, kernel_size=(5, 5), activation='relu', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1164)(x)
        x = Dropout(0.5)(x)
        x = Dense(100)(x)
        x = Dropout(0.5)(x)
        x = Dense(50)(x) 
        x = Dropout(0.2)(x)
        x = Dense(10)(x)

        B_zeta_model = tf.keras.models.Model(
              inputs = [Input], outputs = [x])
        # keras model for basis functions B_zeta
        
        print('loading model weights ...')
        if self.model == 'precise':
            checkpoint_path = '../../../../data/models/20201027_filtered_gaussian_resampled/export/'
            B_zeta_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)) # tf.train.latest_checkpoint(checkpoint_path)
            self.Bzetamodel = B_zeta_model
            print('... finished loading weights.')
            
        elif self.model == 'imprecise':
            checkpoint_path = '../../../data/models/20201021_unrestr_gaussian_resampled/export/'
            B_zeta_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)) # tf.train.latest_checkpoint(checkpoint_path)
            self.Bzetamodel = B_zeta_model
            print('... finished loading weights.')
        else:
            return('unknown model type')
    
    def load_z_model(self):
        
        # define model and load weights from training
        Input = tf.keras.layers.Input(shape=(66, 200, 3,), name='image')
        x = Conv2D(24, kernel_size=(5, 5), activation='relu', strides=(2, 2))(Input)
        x = BatchNormalization()(x)
        x = Conv2D(36, kernel_size=(5, 5), activation='relu', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(48, kernel_size=(5, 5), activation='relu', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1164)(x)
        x = Dropout(0.5)(x)
        x = Dense(100)(x)
        x = Dropout(0.5)(x)
        x = Dense(50)(x) 
        x = Dropout(0.2)(x)
        x = Dense(10)(x)
        Output = Dense(1, name = 'output_layer')(x)

        z_model = tf.keras.models.Model(
              inputs = [Input], outputs = [Output])
        
        print('loading model weights ...')
        if self.model == 'precise':
            checkpoint_path = '../../../../data/models/20201027_filtered_gaussian_resampled/'
            z_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)) # tf.train.latest_checkpoint(checkpoint_path)
            self.z_model = z_model
            print('... finished loading weights.')
            
        elif self.model == 'imprecise':
            checkpoint_path = '../../../data/models/20201021_unrestr_gaussian_resampled/'
            z_model.load_weights(tf.train.latest_checkpoint(checkpoint_path)) # tf.train.latest_checkpoint(checkpoint_path)
            self.z_model = z_model
            print('... finished loading weights.')
        else:
            return('unknown model type')
    
    def predict_z(self, image_path_list):
        
        print('start predicting z on images ...')
        z_preds = []
        images = []
        
        for img_path in image_path_list:
            
            # load image
            img = imageio.imread(img_path)
            images.append(img)
            #img = cv2.resize(img, dsize = (291,218), interpolation = cv2.INTER_LINEAR)[76:142, 45:245,0:3].reshape(1,66,200,3)/255
            img = img[:,:,0:3].reshape(1,66,200,3)/255
            # predict Bzeta
            z_pred = self.z_model.predict(img)
            z_preds.append(z_pred)
        
        print('... finished predicting z on images.')
        self.z_pred = np.array(z_preds).reshape(len(image_path_list),)
        self.images = images
        return(self.z_pred)
        
        
    
    def predict_Bzeta(self, image_path_list, ifpath, single_img):
        
        print('start predicting Bzeta on images ...')
        B_zetas = []
        images = []
        
        if ifpath:
            for img_path in image_path_list:
                # load image
                img = imageio.imread(img_path)
                images.append(img)
                #img = cv2.resize(img, dsize = (291,218), interpolation = cv2.INTER_LINEAR)[76:142, 45:245,0:3].reshape(1,66,200,3)/255
                img = img[:,:,0:3].reshape(1,66,200,3)/255
                # predict Bzeta
                B_zeta = self.Bzetamodel.predict(img)
                B_zetas.append(B_zeta)
        
        else:
            B_zeta = self.Bzetamodel.predict(single_img)
            B_zetas.append(B_zeta)
            
        
        print('... finished predicting Bzeta on images.')
        self.B_zetas = np.array(B_zetas).reshape(len(image_path_list), self.p)
        self.images = images
    
    def choose_method(self, method):
        '''choose desired estimation method, choice of:
            - normal DNN: 'dnn'
            - Ridge prior & HMC: 'hmc_ridge'
            - Ridge prior & VA: 'va_ridge
            - Horseshoe prior & HMC: 'hmc_horseshoe'
            - Horseshoe prior & VA: 'va_horseshoe'
            '''
        
        self.method = method
        if method == 'dnn':
            beta = np.genfromtxt(str(self.extracted_coefficients_path_beta + 'beta.csv'), delimiter = ',')
        
        elif method == 'hmc_ridge':
            self.hmc_ridge_dir = '../../../../data/commaai/mcmc/filtered_gaussian_resampled/Ridge/'
            self.mu_t_hmc = np.load(self.hmc_ridge_dir + 'all_thetas.npy')[500:,:]
            self.beta = np.mean(self.mu_t_hmc[:,0:10], axis = 0)
            self.tau_sq = np.exp(self.mu_t_hmc[:,10])

        elif (method == 'va_ridge') & (self.model == 'precise'):
            self.va_ridge_dir = '../../../../data/commaai/va/filtered_gaussian_resampled/Ridge/'
            self.mu_t_va = np.load(self.va_ridge_dir + 'mu_ts.npy')
            self.iter = self.mu_t_va.shape[0]
            self.beta = np.mean(self.mu_t_va[int(0.95*self.iter):self.iter, 0:10], axis = 0)
            self.tau_sq = np.exp(np.mean(self.mu_t_va[int(0.95*self.iter):self.iter, 10], axis = 0))
            print('chose variational approximation with ridge prior as method')
            
        elif (method == 'hmc_horseshoe') & (self.model == 'precise'):
            self.hmc_ridge_dir = '../../../../data/commaai/mcmc/filtered_gaussian_resampled/Horseshoe/'
            self.mu_t_hmc = np.load(self.hmc_ridge_dir + 'all_thetas.npy').reshape(-1, 21)
            self.beta = np.mean(self.mu_t_hmc[15000:,0:10], axis = 0)
            self.Lambda = np.exp(0.5*self.mu_t_hmc[15000:,10:20])
            self.tau_sq = np.exp(self.mu_t_hmc[15000:,20])
            print('chose hmc with horseshoe prior as method')
        
        elif (method == 'va_ridge') & (self.model == 'imprecise'):
            va_ridge_dir = '../../../../data/commaai/va/unfiltered_gaussian_resampled/Ridge/'
            self.mu_t_va = np.load(va_ridge_dir + 'mu_ts.npy')
            self.iter = self.mu_t_va.shape[0]
            self.beta =  np.mean(self.mu_t_va[int(0.95*self.iter):self.iter,0:10], axis = 0)
            self.tau_sq = np.exp(np.mean(self.mu_t_va[int(0.95*self.iter):self.iter,10], axis = 0))
            print('chose variational approximation with ridge prior as method')
        
        elif (method == 'va_horseshoe') & (self.model == 'precise'):
            self.va_horseshoe_dir = '../../../data/commaai/va/filtered_gaussian_resampled/Horseshoe/'
            self.mu_t_va = np.load(self.va_horseshoe_dir + 'mu_ts.npy')
            self.iteration = self.mu_t_va.shape[0]
            self.p = 10
            self.B_ts = np.mean(np.load(self.va_horseshoe_dir + 'B_ts.npy')[int(0.95*self.iteration):,:,:], axis = 0)
            self.d_ts = np.mean(np.load(self.va_horseshoe_dir + 'd_ts.npy')[int(0.95*self.iteration):,:,:], axis = 0)
            self.var = np.sqrt(np.diag(self.B_ts.dot(self.B_ts.T) + self.d_ts**2))
            self.beta = np.mean(self.mu_t_va[int(0.9*self.iteration):,0:10], axis = 0)
            self.Lambdas_log = np.mean(self.mu_t_va[int(0.9*self.iteration):,10:20], axis = 0)
            self.samples = np.exp(0.5*np.random.multivariate_normal(self.Lambdas_log.reshape(10,), np.diag(self.var[10:20]), 10000))
            self.Lambda = np.mean(self.samples, axis = 0)
            print('chose variational approximation with horseshoe prior as method')
    
    def initialize_grid(self, density, no_points):
        
        print('computing fixed values for density estimation ...')
        self.grid = np.linspace(min(density['axes']), max(density['axes']), no_points)
        density_y = density['axes']
        density_pdf = density['pdf']
        # compute these beforehand to save computation time
        self.p_y_y0 = [density_pdf[find_closest_element(y_i,density_y)] for y_i in self.grid]
        self.part_1 = np.array([norm.ppf(Fy(y_i, density)) for y_i in self.grid])
        self.phi_1_z = np.array([scipy.stats.norm(0, 1).pdf(y_i) for y_i in self.part_1 ])
        print('...finished computing fixed values.')
        
    def predict_density(self):
        
        def predict_single_density(x, grid, p_y_y0, part_1, phi_1_z, beta, tau_sq, Lambda, method):
    
            psi_x0 = x

            f_eta_x0 = psi_x0.dot(beta)
            
            if method == 'va_ridge':
                s_0_hat = (1 + tau_sq*psi_x0.dot(psi_x0))**(-0.5)
                
            if method == 'hmc_ridge':
                s_0_hats =  []
                for tau_j in tau_sq:
                    s_0_hatj = (1 + tau_j*psi_x0.dot(psi_x0))**(-0.5)
                    s_0_hats.append(s_0_hatj)
                s_0_hat = np.mean(np.array(s_0_hats))
            
            elif method == 'va_horseshoe':
                s_0_hat = (1 + (psi_x0*(Lambda**2)).dot(psi_x0))**(-0.5)
                
            elif method == 'hmc_horseshoe':
                s_0_hats =  []
                for Lambda_j in self.Lambda:
                    s_0_hatj = (1 + (psi_x0*(Lambda_j**2)).dot(psi_x0))**(-0.5)
                    s_0_hats.append(s_0_hatj)
                s_0_hat = np.mean(np.array(s_0_hats))

            part_0 = s_0_hat*f_eta_x0
            
            # compute the cdf of new ys
            term_1 = scipy.stats.norm(0, 1).pdf((part_1 - part_0) / s_0_hat)
            p_y_single_obs_whole_dens = (p_y_y0/phi_1_z)*(1/s_0_hat)*term_1

            return(p_y_single_obs_whole_dens)
        
        if self.method == 'va_ridge':
            self.Lambda = np.zeros(1)
            self.densities = [predict_single_density(self.B_zetas[i,:], self.grid, self.p_y_y0, self.part_1, self.phi_1_z, self.beta, self.tau_sq, self.Lambda, self.method) for i in range(0,self.B_zetas.shape[0])]
            #densities = np.array(densities).reshape(self.B_zetas.shape[0], len(self.grid))
        if self.method == 'va_horseshoe':
            self.tau_sq = np.zeros(1)
            self.densities = [predict_single_density(self.B_zetas[i,:], self.grid, self.p_y_y0, self.part_1, self.phi_1_z, self.beta, self.tau_sq, self.Lambda, self.method) for i in range(0,self.B_zetas.shape[0])]
        
        if self.method == 'hmc_horseshoe':
            self.tau_sq = np.zeros(1)
            self.densities = [predict_single_density(self.B_zetas[i,:], self.grid, self.p_y_y0, self.part_1, self.phi_1_z, self.beta, self.tau_sq, self.Lambda, self.method) for i in range(0,self.B_zetas.shape[0])]
            
        if self.method == 'hmc_ridge':
            self.tau_sq = np.zeros(1)
            self.Lambda = np.zeros(1)
            self.densities = [predict_single_density(self.B_zetas[i,:], self.grid, self.p_y_y0, self.part_1, self.phi_1_z, self.beta, self.tau_sq, self.Lambda, self.method) for i in range(0,self.B_zetas.shape[0])]
        return(self.densities)
    
    def show_predictive_density(self, image_number, label, pred):
        
        ang_dens = pd.DataFrame({'angle': np.linspace(-50,50,1000), 'density': self.densities[image_number]})
        max_dens = max(ang_dens['density'])
        const = 1/max_dens
        def plot_density(ang_dens, c, r):
            ang_dens['angle_rad'] = ang_dens['angle'].apply(lambda x: x / 180. * np.pi + np.pi / 2)
            ang_dens['t'] = ang_dens['angle_rad'].apply(lambda x: (c[0] + np.cos(x) * r, c[1] - np.sin(x) * r))
            for i in range(0, ang_dens['t'].shape[0]):
                t = ang_dens['t'][i]
                dens = max(0,ang_dens['density'][i]*const)
                plt.plot((c[0], t[0]), (c[1], t[1]), 'dodgerblue', alpha = dens)

        figure, ax = plt.figure(), plt.gca()
        ax.imshow(self.images[image_number][:,:,0:3].astype(int))
        a_rad = label / 180. * np.pi + np.pi / 2
        c, r = (582,873), 300 #center, radius
        plot_density(ang_dens, c, r)
        t = (c[0] + int(np.cos(a_rad) * r), c[1] - int(np.sin(a_rad) * r))
        plt.plot((c[0], t[0]), (c[1], t[1] + 30), 'w', alpha = 1)
        a_rad = pred / 180. * np.pi + np.pi / 2
        c, r = (582,873), 300 #center, radius
        t = (c[0] + int(np.cos(a_rad) * r), c[1] - int(np.sin(a_rad) * r))
        plt.plot((c[0], t[0]), (c[1] , t[1] + 30), 'r', alpha = 1)
        plt.gca().add_artist(plt.Circle(c, r, edgecolor='dodgerblue', facecolor='k'))
        plt.text(c[0] - 550 , c[1] - 800, 'steering angle'.format(label), color='w')
        plt.text(c[0] - 550 , c[1] - 750, 'true: {:0.1f}$^\circ$'.format(label), color='w')
        plt.text(c[0] - 550 , c[1] - 700, 'predicted: {:0.1f}$^\circ$'.format(pred), color='r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        figure.savefig('example_steering_visualization.pdf', format='pdf', dpi=900)
    
    
    def confidence_interval(self, density, confidence_level, z_preds):
        
        confidence_intervals = []
        j = 0
        for z_pred in z_preds:
            pred_y = density.loc[find_closest_element(norm.cdf(z_pred), density['cdf']), 'axes']

            integrals = []

            for i in range(0, len(densities[0]) - 1):
                integral = integrate.simps([max(0,self.densities[0][i]),max(0,self.densities[0][i + 1])], [self.grid[i], self.grid[i+1]])
                integrals.append(integral)
                
            int_sum = 0
            i = 1
            start_index = np.searchsorted(density['axes'], pred_y)
            while int_sum < confidence_level:
                int_sum += integrals[start_index - i] + integrals[start_index + i]
                i += 1
            confidence_intervals.append([density.loc[start_index - i,'axes'], density.loc[start_index + i,'axes']])
        return(confidence_intervals)
    

def build_model_mc_dropout():
    '''
    Build PilotNet keras model
    
    Input:
        None
    
    Output:
        - keras_model: keras model with PilotNet architecture
    '''
    
    Lambda = 10**(-6)
    # build model
    Input = tf.keras.layers.Input(shape=(66, 200, 3,), name='image')
    x = Conv2D(24, kernel_size=(5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(Input)
    x = Dropout(.05)(x, training = True)
    #x = BatchNormalization()(x)
    x = Conv2D(36, kernel_size=(5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(x)
    x = Dropout(.05)(x, training = True)
    #x = BatchNormalization()(x)
    x = Conv2D(48, kernel_size=(5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(x)
    x = Dropout(.05)(x, training = True)
    #x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(x)
    #x = BatchNormalization()(x)
    x = Dropout(.05)(x, training = True)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda))(x)
    x = Flatten()(x, training = True)
    x = Dropout(0.05)(x, training = True)
    x = Dense(1164, kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda), activation='relu')(x)
    x = Dropout(0.05)(x, training = True)
    x = Dense(100, kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda), activation='relu')(x)
    x = Dropout(0.05)(x, training = True)
    x = Dense(50, kernel_regularizer=l2(Lambda), bias_regularizer=l2(Lambda), activation='relu')(x) 
    x = Dropout(0.05)(x, training = True)
    x = Dense(10, activation='relu')(x)
    Output = Dense(1, name = 'output_layer')(x)

    # compile
    keras_model = tf.keras.models.Model(
          inputs = [Input], outputs = [Output])
    
    return(keras_model)


def build_model_mdn():
    '''
    Build PilotNet keras model
    
    Input:
        None
    
    Output:
        - keras_model: keras model with PilotNet architecture
    '''

    # build model
    Input = tf.keras.layers.Input(shape=(66, 200, 3,), name='image')
    x = Conv2D(24, kernel_size=(5, 5), activation='relu', strides=(2, 2))(Input)
    x = BatchNormalization()(x)
    x = Conv2D(36, kernel_size=(5, 5), activation='relu', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, kernel_size=(5, 5), activation='relu', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1164, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu')(x) 
    x = Dropout(0.2)(x)
    x = Dense(10, activation='relu')(x)
    Output = mdn.MDN(1, 50)(x)

    # compile
    keras_model = tf.keras.models.Model(
          inputs = [Input], outputs = [Output])
    
    return(keras_model)


    