#########################################################
# NLPSupportVectorMachine:
#   SVM for NLP Classification of Spam Email Messages
## SVM (Full Implementation)
##     ++ Hard/Soft Margin, Linear/Polynomial/Gaussian Kernel   
## SVM Helper Functions + PCA + LDA (Full Implementation)
## Author: Samyuel Danyo
## Date: 8/4/2019
##  License: MIT License
##  Copyright (c) 2020 Samyuel Danyo
#########################################################

# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import time
import cvxopt
import cvxopt.solvers
# Set the seed of the numpy random number generator
np.random.seed(seed=1)
from sklearn import model_selection, metrics
import scipy.io
import warnings

# Preprocessing Functions
def standardize(X):
    """ Transform the data X to have a mean of zero
        and a standard deviation of 1."""
    X_mean = np.mean(X, axis=0)
    X_var = np.std(X, axis=0)
    return (X-X_mean)/X_var

def normalize(X):
    """ Rescale the data X to ∈ [0:1]."""
    X_min = X.min()
    X_max = X.max()
    return (X - X_min)/(X_max - X_min)

def log_transform(x):
    """Transform x to natural_log(x)."""
    return np.log(x+0.1)

def binary_transform(x):
    """Transform x to binary based on sign."""
    return np.where(x>0, 1, 0)
	
# Shared Helper Functions For Metrics Extraction & Cofusion Table Plot
def get_accuracy(targets, predictions):
    """ Helper Function for calculating the (%) accuracy of
        'predictions' compared to 'targets'."""
    return (np.abs(targets - predictions) < 1e-10 ).mean() * 100.0

def get_errors(outputs, targets):
    """ Helper Function for calculating the error rates of 'outputs' compared to 'targets'.
        Returns:
            error (NumPy Array): overall error rate.
           false_negatives_error (NumPy Array): the % of samples wrongly outputed as '0' instead of '1'.
           false_positives_error (NumPy Array): the % of samples wrongly outputed as '1' instead of '0'."""
    error_per_sample = []
    error = []
    false_negatives_error = []
    false_positives_error = []
    # Calculation per epoch.
    if(np.array(outputs).ndim == 2):
        for idx, outputs_per_par in enumerate(outputs):
            error_per_sample.append(list(np.array(outputs_per_par) - np.array(targets)))
            error.append(sum(abs(i) for i in error_per_sample[idx])/len(error_per_sample[idx])*100) 
            false_negatives_error.append(abs(sum(i for i in error_per_sample[idx] if i < 0)))/len(error_per_sample)*100
            false_positives_error.append(sum(i for i in error_per_sample[idx] if i > 0))/len(error_per_sample)*100
    # Calcualtion for a single epoch.
    elif(np.array(outputs).ndim == 1):
        error_per_sample = (list((np.array(outputs) - np.array(targets))/2))
        error = (sum(abs(i) for i in error_per_sample)/len(error_per_sample)*100) 
        false_negatives_error = (abs(sum(i for i in error_per_sample if i < 0)))/len(error_per_sample)*100
        false_positives_error = (sum(i for i in error_per_sample if i > 0))/len(error_per_sample)*100
    return error, false_negatives_error, false_positives_error

def plot_confusion_table(y_true, y_pred, title):
    """ Helper Function for displaying a confusion table of targets vs predictions."""
    # Show confusion table
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=None)  # Get confustion matrix
    # Plot the confusion table
    class_names = ['${:d}$'.format(x) for x in (-1, 1)]  # Binary class names
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Show class labels on each axis
    ax.xaxis.tick_top()
    major_ticks = range(0,2)
    minor_ticks = [x + 0.5 for x in range(0, 2)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    # Set plot labels
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted Label', fontsize=15)
    ax.set_ylabel('True Label', fontsize=15)
    fig.suptitle(title, y=1.03, fontsize=15)
    # Show a grid to seperate digits
    ax.grid(b=True, which=u'minor')
    # Color each grid cell according to the number classes predicted
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    # Show the number of samples in each cell
    for x in range(conf_matrix.shape[0]):
        for y in range(conf_matrix.shape[1]):
            color = 'w' if x == y else 'k'
            ax.text(x, y, conf_matrix[y,x], ha="center", va="center", color=color)       
    plt.show()
	
# PCA Transform Class
class PrincipalComponentAnalysis(object):
    """ Principal Component Analysis pattern recognition algorithm.
        PCA aims to find the directions (components) that maximize the variance of the dataset.
        PCA projects a feature space onto a subspace with minimum information loss. """
    
    def __init__(self):
        """Declare the dataset field."""
        self.Components = None
    
    def _check_eigen(self, eigen_val, eigen_vec, cov_mat):
        """ Check if the eigenvalues/eigenvectors equation Σv=λv is satisfied. """
        cov_eigen_vec = cov_mat.dot(eigen_vec)
        eigen_val_vec = eigen_val*eigen_vec
        np.testing.assert_array_almost_equal(cov_eigen_vec, eigen_val_vec,
                                             decimal=6, err_msg='Eigen Vector/Value Equation NOT Satisfied!', verbose=True)
    
    def extract(self, X):
        """ Find the components of the dataset."""
        cov_mat = np.cov(X.T)
        
        # Singular Value Decomposition (SVD)
        eigen_val, eigen_vec = np.linalg.eig(cov_mat)
        self._check_eigen(eigen_val, eigen_vec, cov_mat)
        
        # Get the components - a list of (eigenvalue, component) tuples
        self.Components = [(np.abs(eigen_val[f]), eigen_vec[:,f]) for f in range(len(eigen_val))]

        # Sort the components from high to low
        self.Components.sort(key=lambda x: x[0], reverse=True)
    
    def project(self, X, dims=1):
        """ Project the input X into <dims>-dimensional subspace, based on the
            extracted components. """
        W = self.Components[0][1].reshape(self.Components[0][1].shape[0],1)
        for dim in range(1,dims):
            W = np.hstack((W, self.Components[dim][1].reshape(self.Components[dim][1].shape[0],1)))
        Y = X.dot(W)
        return (Y, W)
		
# LDA Transform Class
class LinearDiscriminantAnalysis(object):
    """ Linear Discriminant Analysis pattern recognition algorithm.
        LDA aims to find the directions that maximize the variance (separation)
        between classes & minimize the variance (difference) inside the classes
        (represented by maximizing the Fisher’s ratio: max( (m1-m2)**2 / (v1+v2) ).
        LDA projects an input data space onto a sub-feature space with
        minimum class-discriminatory information loss. """
    
    def __init__(self):
        """Declare the dataset field."""
        self.Directions = None
    
    def _check_eigen(self, eigen_val, eigen_vec, cov_mat):
        """ Check if the eigenvalues/eigenvectors equation Σv=λv is satisfied. """
        cov_eigen_vec = cov_mat.dot(eigen_vec)
        eigen_val_vec = eigen_val*eigen_vec
        np.testing.assert_array_almost_equal(np.abs(cov_eigen_vec), np.abs(eigen_val_vec),
                                             decimal=1, err_msg='Eigen Vector/Value Equation NOT Satisfied!', verbose=True)
    
    def extract(self, X, T):
        """ Extract the dataset (X, dependent on the class T) directions."""
        class_labs = np.unique(T)
        dim = X.shape[1]
        X_mean = np.mean(X, axis=0).reshape(dim, 1)

        # Calculate the class prior probabilities and means.
        class_priors = []
        class_means = []
        for lab in class_labs:
            X_class = X[T==lab]
            class_priors.append(len(X_class)/len(X))
            class_means.append(np.mean(X_class, axis=0).reshape(dim, 1))
  
        # Build the Within-Class Scatter Matrix (minimizing in-class variance).
        S_W = np.zeros((dim, dim))
        for idx, lab in enumerate(class_labs):
            X_class = X[T==lab]
            class_scatter = np.cov(X_class.T)
            S_W += class_scatter
            
        # Build the Between-Class Scatter Matrix (maximizing between-class separation).
        S_B = np.zeros((dim, dim))
        for idx, lab in enumerate(class_labs):
            S_B += class_priors[idx]*(class_means[idx] - X_mean).dot((class_means[idx] - X_mean).T)
        
        # Singular Value Decomposition (SVD)
        eigen_val, eigen_vec = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        cov_mat = np.linalg.inv(S_W).dot(S_B)
        self._check_eigen(eigen_val, eigen_vec, cov_mat)
        
        # Get the directions - a list of (eigenvalue, direction) tuples
        self.Directions = [(np.abs(eigen_val[f]), eigen_vec[:,f]) for f in range(len(eigen_val))]

        # Sort the directions from high to low meaningfullness
        self.Directions.sort(key=lambda x: x[0], reverse=True)
    
    def project(self, X, dims=1):
        """ Project the input X into <dims>-dimensional subspace, based on the
            extracted directions. """
        # Buld the weight vector (column combination of the <dims> most-meaningfull directions).
        W = self.Directions[0][1].reshape(self.Directions[0][1].shape[0],1)
        for dim in range(1,dims):
            W = np.hstack((W, self.Directions[dim][1].reshape(self.Directions[dim][1].shape[0],1)))
        # Project
        Y = X.dot(W)
        return (Y, W)
    
def visualize_LDA(X, T, set_n = 'Test'):
    """ Helper Function for visualizing the feature space."""
    # Seperate the classes.
    X1 = X[T == -1]
    X2 = X[T == 1]
    dims = X.shape[1]
    
    # 2-D Data
    if dims == 2:
        plt.plot(X1[:,0], X1[:,1], 'o', markersize=7, color='blue', alpha=0.5, label='Not-SPAM')
        plt.plot(X2[:,0], X2[:,1], '^', markersize=7, color='red', alpha=0.5, label='SPAM')
        plt.xlabel('LDA_X1', fontsize=12)
        plt.ylabel('LDA_X2', fontsize=12)
        plt.legend()
        plt.title('{} Dataset LDA-Projected Into 2-D Space'.format(set_n))
        plt.show()
    # #-D Data
    elif dims == 3:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 10   
        ax.plot(X1[:,0], X1[:,1], X1[:,2], 'o', markersize=8, color='blue', alpha=0.5, label='Not-SPAM')
        ax.plot(X2[:,0], X2[:,1], X2[:,2], '^', markersize=8, alpha=0.5, color='red', label='SPAM')
        ax.set_xlabel('LDA_X1', fontsize=12)
        ax.set_ylabel('LDA_X2', fontsize=12)
        ax.set_zlabel('LDA_X3', fontsize=12)
        plt.title('{} Dataset LDA-Projected Into 3-D Space'.format(set_n))
        ax.legend()
        plt.show()
		
# Kernel Functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
	
# SVM Class
class SupportVectorMachine(object):
    """ Support Vector Machine classifier implementation.
            - Kernel trick.
            - Hard margin fit.
            - Soft margin fit.
            
        The SVM classifier aims to find the optimal hyperplane decision boundary (separation) between two classes {-1,1}.
        It approaches the task by maximizing the margin (distance) between the data and the hyperplane
        (the hyperplane can be shifted to make sure it separates the margin equally). The vectors going through
        the points between which and the hyperplane is the margin are called support vectors.
        Maximizing the margin is proven to yield the optimal separation (leads to lower probability of misclassification)
        by Cristianini and Shawe-Taylor. This leads to a optimization problem, where we try to maximize the normalized
        (geometric) margin, while keeping the absolute (functional) margin the same (=1), which leads to minimizing the weights
        (hyperplane parameters). The problem can be summed up as minimizing 1/2W.T*W, subject to d*(W.T*X+b)>=1 (d = class).
        Such primal optimization problem can be approached, using Lagrange Theorem, aiming to find the equality and inequality
        Lagrange multipliers. The optimization problem can be further remodeled, using the Kuhn-Tucker Theorem,
        to transform it from primal to dual problem, where a single parameter yields the solution to the Lagrange multipliers.
        In order to solve the problem, dual quadratic programming is required, which yields the Lagrange multipliers and hence
        the hyperplane parameters.
        The obvious problem though is that this is a linear separation. Meaning it will work for linearly separable classes.
        In order to improve the performance on overlapping data (which still is more or less linear), the soft margin is introduced.
        The concept allows for data-points to be inside the margin (basically ignoring them, when calculating the margin),
        by introducing the slack variables, which added together and scaled by ‘C’ (the cost of violating the constraints)
        are the error penalty. The optimization problem now aims not only to maximize the margin but also minimize the error.
        The third SVM concept is the kernel transform (known as the Kernel trick).
        It is based on Cover's Theorem, which states that: “Probability that classes are linearly separable increases
        when data points in input space are nonlinearly mapped to a higher dimensional feature space.”.
        In order to deal with non-linear data (which is not separable in the data space), the kernel transform K(),
        maps the input data into a feature space, before extracting the support vectors. """
    
    def __init__(self, kernel=linear_kernel, margin_type = 'hard', C=None, p=None):
        """ Initialize the classifier fields.
            Args:
                kernel      (Function): Kernel data transformation function.
                margin_type (String): The SVM margin type. Hard or Soft.
                C           (Float): The soft margin cost of violating constraints. """
        # Kernel
        self.kernel = kernel
        self.p = None
        if kernel is polynomial_kernel:
            self.p = int(p)
        # Margin
        if margin_type not in ('hard', 'soft'):
            raise ValueError('Please Choose Valid Margin Type: (hard, soft)')
        self.margin_type = margin_type
        # Cost of Violating Constraints
        self.C = None
        if margin_type is 'soft':
            self.C = float(C)
        # Hyperplane Parameters
        self.W = None
        self.b = None
        # Lagrange Multipliers
        self.a = None
        # Support Vectors
        self.SV = None
        # Support Vectors Targets
        self.SV_T = None
        
    def _get_gram_matrix(self, X):
        """ Computing the Gram Matrix of the
            transofrmed input data.
            Args:
                X (NumPy Array): Dataset Samples. """
        n_samples = X.shape[0]
        GM = np.zeros((n_samples, n_samples))
        if self.kernel is polynomial_kernel:
            for i in range(n_samples):
                for j in range(n_samples):
                    GM[i,j] = self.kernel(X[i], X[j], p=self.p)
        else:
            for i in range(n_samples):
                for j in range(n_samples):
                    GM[i,j] = self.kernel(X[i], X[j])
        return GM
    
    def _check_gram_matrix(self, GM):
        """ Check if the Gram matrix satisfies the
            Mercer Condition (if the kernel is valid). """
        eigen_val, _ = np.linalg.eig(GM)
        if np.any(eigen_val) < 0:
            raise ValueError("Kernel Does NOT Satisfy Mercer's Condition!")

    def fit(self, X, T):
        """ Find the optimal decision boundary (Hyperplane) based on the SVs.
            Args:
                X (NumPy Matrix): Input samples.
                T (NumPy Vector): Input samples targets.
            Components:
                Solve pair of primal and dual convex quadratic programs (optimisation problems).
                x,y - Optimisation Variables, Targets
                P,q - Define any quadratic objective function of x
                G,h - Define inequality constraints
                A,b - Define equality constraints
                <<Primal>>
                    minimize:   (1/2)x.TPx + q.Tx (F(w))
                    subject to: Gx <= h           (Q(w))
                                Ax = b            (H(w))
                    Generalized Lagrangian Function (L(w,a,β)):
                                F(w) + aQ(w) + βH(w)
                <<Dual>>
                    maximize:   -(1/2)(q + G.Ta + A.Ty).TP(q + G.Ta + A.Ty) - h.Ta -b.Ty (L(w,a,β))
                    subject to: q + G.Ta + A.Ty ∈ range(P)                               (∂L(w,a,β)/∂w = 0)
                                (C >=) a >= 0
                More info: <https://scaron.info/blog/quadratic-programming-in-python.html>
                           <https://cvxopt.org/userguide/coneprog.html>
          """ 
        n_samples, n_features = X.shape
        
        # Gram matrix
        GM = self._get_gram_matrix(X)
        self._check_gram_matrix(GM)
        
        # Coefficients of the Kernel Transform for the dual quadratic program problem.
        P = cvxopt.matrix(np.outer(T,T) * GM)
        # Transposed Coefficients of the input (x) (sign).
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # Equality constraints (coefficients of the constraint equations). 
        A = cvxopt.matrix(T, (1,n_samples))
        b = cvxopt.matrix(0.0)

        # Inequality constraints (coefficients of the constraint equations).
        # Any constraints that are (>=) must be multiplied by -1 to become (<=).
        if self.C is None:
            # Hard margin (a >= 0).
            G = cvxopt.matrix(np.eye(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            # Soft margin (C >= a >= 0). Since Gx <= h does not have a lower bound, it is
            # represented by: ((-1 0), (0 -1), (1 0), (0 1)) * (a a) <= (0 0 C C)
            G_0 = (np.eye(n_samples)*-1)
            G_C = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_0, G_C)))
            h_0 = np.zeros(n_samples)
            h_C = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((h_0, h_C)))

        # Solve the QP problem.
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Get the Lagrange multipliers.
        a = np.ravel(solution['x'])

        # Get the support vectors (they have non zero lagrange multipliers).
        sv = a > 1e-5
        sv_idx = np.arange(len(a))[sv]
        self.a = a[sv]
        self.SV = X[sv]
        self.SV_T = T[sv]
        print("{} Support Vectors Found Out of {} Data-Points".format(len(self.a), n_samples))

        # Get the optimal hyperplane parameters.
        # Intercept (based on the Discriminant Function g(x) solutions (T). b = g(x) - aTK(x,x)
        self.b = 0
        for idx, a in enumerate(self.a):
            self.b += self.SV_T[idx]
            self.b -= np.sum(self.a * self.SV_T * GM[sv_idx[idx],sv])
        # Normilise
        if len(self.a) > 1:
            self.b /= len(self.a)

        # Weight Vector W = aTSV (only for linear kernel, since non-linear depends on (y - predict samples)).
        if self.kernel is linear_kernel:
            self.W = np.zeros(n_features)
            for idx, a in enumerate(self.a):
                self.W += a * self.SV_T[idx] * self.SV[idx]

    def project(self, X):
        """ Function to project X based on the fitted SVM. """
        # Linear kernel
        if self.W is not None:
            return np.dot(X, self.W) + self.b
        # Non-linear kernel
        else:
            if self.kernel is polynomial_kernel:
                Y = np.zeros(len(X))
                for idx, x in enumerate(X):
                    Wx = 0
                    for a, SV_T, SV in zip(self.a, self.SV_T, self.SV):
                        Wx += a * SV_T * self.kernel(x, SV, p=self.p)
                    Y[idx] = Wx
            else:
                Y = np.zeros(len(X))
                for idx, x in enumerate(X):
                    Wx = 0
                    for a, SV_T, SV in zip(self.a, self.SV_T, self.SV):
                        Wx += a * SV_T * self.kernel(x, SV)
                    Y[idx] = Wx
            return Y + self.b

    def predict(self, X):
        """ Discriminative Function to return predictions based on the SVM projection. """
        return np.sign(self.project(X))
		
# SVM Helper Functions
def plot_non_lin_SVM(X_train, T_train, SVM):
    """ Visualise the 2,3-D Non-Linear Kernel SVM fit margin. """
    
    dims = X_train.shape[1]
    X1_train = X_train[T_train==1]
    X2_train = X_train[T_train==-1]
    x0 = (np.min(X_train[:,0])-1)
    y0 = (np.max(X_train[:,1])+1)
    
    if dims == 2:
        plt.plot(X1_train[:,0], X1_train[:,1], "ro", label='Class 1')
        plt.plot(X2_train[:,0], X2_train[:,1], "bo", label='Class 2')
        plt.scatter(SVM.SV[:,0], SVM.SV[:,1], s=100, c="g")
    
        X1, X2 = np.meshgrid(np.linspace(x0,y0,50), np.linspace(x0,y0,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Y = SVM.project(X).reshape(X1.shape)
        plt.contour(X1, X2, Y, [0.0], colors='k', linewidths=1, origin='lower')
        plt.contour(X1, X2, Y + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X1, X2, Y - 1, [0.0], colors='grey', linewidths=1, origin='lower')
        
        plt.xlabel('X1', fontsize=15)
        plt.ylabel('X2', fontsize=15)
        plt.legend()
        plt.title('Non-Linear 2-D SVM Fit')
        plt.axis("tight")
        plt.show()

def show_results_SVM(svm, X, accuracy_train, accuracy_test, false_neg_train,
                     false_pos_train, false_neg_test, false_pos_test, fit_time):
    """ Helper Function for displaying the fit profiling.
        Args:
            svm        (SupportVectorMachine Object): SVM Fitted Classifier.
            X          (NumPy Matrix): Dataset (usually test).
            accuracy   (Float): Prediction Accuracy."""
    dims = X.shape[1]
    #DISPLAY DATA
    m, s = divmod(fit_time, 60)
    h, m = divmod(m, 60)
    print("================================================================================")
    print("OVERALL TIME FOR FIT: {}h:{}m:{:.5f}s".format(h,m,s))
    print("DATA DIMENSIONALITY: {}-D".format(dims))
    print("TRAIN ACCURACY: {:.2f}% | TRAIN ERROR RATE: {:.2f}%".format(accuracy_train, 100-accuracy_train))
    print("  TRAIN FALSE NEGATIVES: {:.2f}% | TRAIN FALSE POSTIVES: {:.2f}%".format(false_neg_train, false_pos_train))
    print("TEST ACCURACY: {:.2f}% | TEST ERROR RATE: {:.2f}%".format(accuracy_test, 100-accuracy_test))
    print("  TEST FALSE NEGATIVES: {:.2f}% | TEST FALSE POSTIVES: {:.2f}%".format(false_neg_test, false_pos_test))
    print("================================================================================")
    
def fit_SVM(X_train, T_train, X_test, T_test, kernel = linear_kernel,
            margin_type = 'hard', C=None, p = None, transform = 'None', pp_dims = 3, plot_lda = False):
    """ A Wrapper Function which encapsulates SVM fitting, testing, data gathering & data display.
        Args:
            X (NumPy Matrix): Dataset samples.
            T (NumPy Vector): Dataset targets."""
    
    start_time = time.time()
    
    # INITIALISE THE SVM classifier
    SVM = SupportVectorMachine(kernel=kernel, margin_type=margin_type, C=C, p=p)
    print("\n###############################################################################")
    print("SVM FIT | KERNEL={} | p={}".format(SVM.kernel, SVM.p))
    print("        | MARGIN_TYPE={} | C={} | TRANSFORM={}-D {}".format(SVM.margin_type, SVM.C, pp_dims, transform))
    print("###############################################################################")
    
    # Copy the sets. If transform is applied, we do not want to change the original sets.
    X_train_in = np.copy(X_train)
    X_test_in = np.copy(X_test)
    
    # TANSFORM DATASPACE -> NEW FEATURE SPACE
    if transform is "PCA":
        # INITIALISE THE PCA module
        PCA = PrincipalComponentAnalysis()

        # EXTRACT THE TRAIN DATASET COMPONENTS
        PCA.extract(X_train)
        # PROJECT X INTO pp_dims-D SPACE
        # Train
        (X_train_in, _) = PCA.project(X_train, pp_dims)
        X_train_in = X_train_in.astype(float)
        # Test
        (X_test_in, _) = PCA.project(X_test, pp_dims)
        X_test_in = X_test_in.astype(float)
    
    elif transform is "LDA":
        # INITIALISE THE LDA module
        LDA = LinearDiscriminantAnalysis()
    
        # EXTRACT THE TRAIN DATASET DIRECTIONS
        LDA.extract(X_train, T_train)
        # PROJECT X INTO pp_dims-D SPACE
        # Train
        (X_train_in, _) = LDA.project(X_train, pp_dims)
        X_train_in = X_train_in.astype(float)
        if plot_lda:
            visualize_LDA(X_train_in, T_train, set_n = 'Train')
        # Test
        (X_test_in, _) = LDA.project(X_test, pp_dims)
        X_test_in = X_test_in.astype(float)
        if plot_lda:
            visualize_LDA(X_test_in, T_test, set_n = 'Test')
    
    elif transform is "None":
        print("No Transform Will Be Applied.")
    else:
        warnings.warn("Invalid Transform Selected. No Transform Will Be Performed!")
    
    # CHECK IF THE DATASET IS LABELED [-1,1]
    class_labs = np.unique(T_train)
    if len(class_labs) != 2:
        raise ValueError('Please Pass 2-label Dataset For SVM Fitting.')
    elif class_labs is not np.array([-1, 1]):
        T_train_class = np.copy(T_train)
        T_train_class[T_train==class_labs[0]] = -1
        T_train_class[T_train==class_labs[1]] = 1
        T_test_class = np.copy(T_test)
        T_test_class[T_test==class_labs[0]] = -1
        T_test_class[T_test==class_labs[1]] = 1
        
        # FIT SVM
        SVM.fit(X_train_in, T_train_class)
    
        # PREDICT
        Y = SVM.predict(X_train_in)
        accuracy_train = get_accuracy(T_train_class, Y)
        _, false_neg_error_train, false_pos_error_train = get_errors(Y, T_train_class)
        Y = SVM.predict(X_test_in)
        accuracy_test = get_accuracy(T_test_class, Y)
        _, false_neg_error_test, false_pos_error_test = get_errors(Y, T_test_class)
    else:
        # FIT SVM
        SVM.fit(X_train_in, T_train)
    
        # PREDICT
        Y = SVM.predict(X_train_in)
        accuracy_train = get_accuracy(T_train, Y)
        _, false_neg_error_train, false_pos_error_train = get_errors(Y, T_train)
        Y = SVM.predict(X_test_in)
        accuracy_test = get_accuracy(T_test, Y)
        _, false_neg_error_test, false_pos_error_test = get_errors(Y, T_test)
    
    fit_time = time.time() - start_time
    
    # VISUALIZE SOLUTION
    if kernel is not linear_kernel:
        plot_non_lin_SVM(X_train_in, T_train, SVM)
    plot_confusion_table(T_test, Y, "{} Margin {} SVM Confusion Table".format(margin_type, kernel))
    
    # DISPLAY FIT DATA
    show_results_SVM(SVM, X_train_in, accuracy_train, accuracy_test, false_neg_error_train,
                     false_pos_error_train, false_neg_error_test, false_pos_error_test, fit_time)

# Dataset Preprocessing
# Get the Dataset into NumPy arrays.
EMAIL_TRAIN_DICT = scipy.io.loadmat('train.mat')
X_TRAIN = np.array(EMAIL_TRAIN_DICT["train_data"]).T
T_TRAIN = np.array(EMAIL_TRAIN_DICT["train_label"])
T_TRAIN = T_TRAIN.reshape(T_TRAIN.shape[0])

EMAIL_TEST_DICT = scipy.io.loadmat('test.mat')
X_TEST = np.array(EMAIL_TEST_DICT["test_data"]).T
T_TEST = np.array(EMAIL_TEST_DICT["test_label"])
T_TEST = T_TEST.reshape(T_TEST.shape[0])

EMAIL_DATASET = np.column_stack((np.vstack((X_TRAIN, X_TEST)),np.hstack((T_TRAIN, T_TEST))))
print(">>>>>> EMAIL DATASET FEATURE SPACE <<<<<<")
print(">>> FULL <<<")
print(np.array(EMAIL_DATASET).shape)
print(">>> TRAIN <<<")
print(X_TRAIN.shape)
print(T_TRAIN.shape)
print(">>> TEST <<<")
print(X_TEST.shape)
print(T_TEST.shape)

# Preprocess the data
X_TRAIN_BNY = binary_transform(X_TRAIN)
X_TEST_BNY = binary_transform(X_TEST)

X_TRAIN_LOG = log_transform(X_TRAIN)
X_TEST_LOG = log_transform(X_TEST)

X_TRAIN_STD = standardize(X_TRAIN)
X_TEST_STD = standardize(X_TEST)

# Train & Test | Hard Margin SVM With Linear Kernel
# CONTROL HYPERPARAMETERS HARD MARGIN LINEAR KERNEL 
X_train_list = [X_TRAIN_STD, X_TRAIN_LOG]
X_train_list_names = ['X_TRAIN_STD', 'X_TRAIN_LOG']
X_test_list = [X_TEST_STD, X_TEST_LOG]
Transform_list = ['PCA', 'LDA']
Dims_list = [15, 2]

# RUN HARD MARGIN LINEAR KERNEL EXPERIMENT
print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("HARD MARGIN LINEAR KERNEL EXPERIMENT")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("X_train = X_TRAIN_BNY")
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
fit_SVM(X_TRAIN_BNY, T_TRAIN.astype(float), X_TEST_BNY, T_TEST.astype(float),
            kernel = linear_kernel, margin_type = 'hard', transform = 'None', pp_dims = 57)

print("Would you like to run hard margin, linear kernel experiment for data-transforms: STD+PCA, LOG+LDA?")
USER_I = input()
if(USER_I in ('yes', 'Yes', 'Y', 'y')):
	for idx, (X_train, X_test, transform, dims) in enumerate(zip(X_train_list, X_test_list, Transform_list, Dims_list)):
		print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
		print("X_train = {}".format(X_train_list_names[idx]))
		print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		fit_SVM(X_train, T_TRAIN.astype(float), X_test, T_TEST.astype(float),
				kernel = linear_kernel, margin_type = 'hard', transform = transform, pp_dims = dims, plot_lda=True)
			
# Train & Test | Hard Margin SVM With Polynomial Kernel
# CONTROL HYPERPARAMETERS HARD MARGIN POLYNOMIAL KERNEL
X_train_list = [X_TRAIN_STD, X_TRAIN_LOG]
X_train_list_names = ['X_TRAIN_STD', 'X_TRAIN_LOG']
X_test_list = [X_TEST_STD, X_TEST_LOG]
Transform_list = ['PCA', 'LDA']
Dims_list = [15, 2]
P_list = [2,3,4,5]

# RUN HARD MARGIN POLYNOMIAL KERNEL EXPERIMENT
print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("HARD MARGIN POLYNOMIAL KERNEL EXPERIMENT")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("X_train = X_TRAIN_BNY")
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
for p in P_list:
	fit_SVM(X_TRAIN_BNY, T_TRAIN.astype(float), X_TEST_BNY, T_TEST.astype(float),
			kernel = polynomial_kernel, margin_type = 'hard', p=p, transform = 'None', pp_dims = 57)

print("Would you like to run hard margin, polynomial kernel experiment for data-transforms: STD+PCA, LOG+LDA?")
USER_I = input()
if(USER_I in ('yes', 'Yes', 'Y', 'y')):
	for idx, (X_train, X_test, transform, dims) in enumerate(zip(X_train_list, X_test_list, Transform_list, Dims_list)):
		print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
		print("X_train = {}".format(X_train_list_names[idx]))
		print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		for p in P_list:
			fit_SVM(X_train, T_TRAIN.astype(float), X_test, T_TEST.astype(float),
					kernel = polynomial_kernel, margin_type = 'hard', p=p, transform = transform, pp_dims = dims)
				
# Train & Test | Soft Margin SVM With Polynomial Kernel
# CONTROL HYPERPARAMETERS SOFT MARGIN POLYNOMIAL KERNEL
X_train_list = [X_TRAIN_STD, X_TRAIN_LOG]
X_train_list_names = ['X_TRAIN_STD', 'X_TRAIN_LOG']
X_test_list = [X_TEST_STD, X_TEST_LOG]
Transform_list = ['PCA', 'LDA']
Dims_list = [15, 2]
P_list = [1,2,3,4,5]
C_list = [0.1, 0.6, 1.1, 2.1]

# RUN SOFT MARGIN POLYNOMIAL KERNEL EXPERIMENT
print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("SOFT MARGIN POLYNOMIAL KERNEL EXPERIMENT")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("X_train = X_TRAIN_BNY")
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
for p in P_list:
        for C in C_list:
            fit_SVM(X_TRAIN_BNY, T_TRAIN.astype(float), X_TEST_BNY, T_TEST.astype(float),
			        kernel = polynomial_kernel, margin_type = 'soft', C=C, p=p, transform = 'None', pp_dims = 57)

print("Would you like to run soft margin, polynomial kernel experiment for data-transforms: STD+PCA, LOG+LDA?")
USER_I = input()
if(USER_I in ('yes', 'Yes', 'Y', 'y')):
	for idx, (X_train, X_test, transform, dims) in enumerate(zip(X_train_list, X_test_list, Transform_list, Dims_list)):
		print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
		print("X_train = {}".format(X_train_list_names[idx]))
		print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		for p in P_list:
			for C in C_list:
				fit_SVM(X_train, T_TRAIN.astype(float), X_test, T_TEST.astype(float),
						kernel = polynomial_kernel, margin_type = 'soft', C=C, p=p, transform = transform, pp_dims = dims)
					
# EVALUATION | Soft Margin SVM With Gaussian Kernel
print("Would You Like to Perform Evaluation? (y/n)")
USER_I = input()
if(USER_I in ('yes', 'Yes', 'Y', 'y')):
    # Set the Dataset Args
    filename, data_set, lab_set = 'eval.mat', 'eval_data', 'eval_label'
    print("Please Indicate if The Evaluation Dataset Args Are Correct? (y/n)")
    print("Filename:{} | Data Set:{} | Label Set:{}".format(filename, data_set, lab_set))
    USER_I = input()
    if(USER_I not in ('yes', 'Yes', 'Y', 'y')):
        print("Please Enter Filename")
        filename = input()
        print("Please Enter Data Set Variable Name")
        data_set = input()
        print("Please Enter Label Set Variable Name")
        lab_set = input()
    
    # Get the Dataset
    EMAIL_EVAL_DICT = scipy.io.loadmat(filename)
    X_EVAL = np.array(EMAIL_EVAL_DICT[data_set]).T
    T_EVAL = np.array(EMAIL_EVAL_DICT[lab_set])
    T_EVAL = T_EVAL.reshape(T_EVAL.shape[0])
    print(">>>>>> EMAIL EVAL DATASET FEATURE SPACE <<<<<<")
    print(X_EVAL.shape)
    print(T_EVAL.shape)
    
    # Preprocess the Dataset
    X_TRAIN_LOG = log_transform(X_TRAIN)
    X_EVAL_LOG = log_transform(X_EVAL)

    X_TRAIN_BNY = binary_transform(X_TRAIN)
    X_EVAL_BNY = binary_transform(X_EVAL)

    X_TRAIN_STD = standardize(X_TRAIN)
    X_EVAL_STD = standardize(X_EVAL)
    
    # CONTROL HYPERPARAMETERS SOFT MARGIN GAUSSIAN KERNEL
    X_train_list = [X_TRAIN_BNY, X_TRAIN_STD, X_TRAIN_LOG]
    X_train_list_names = ['X_TRAIN_BNY', 'X_TRAIN_STD', 'X_TRAIN_LOG']
    X_eval_list = [X_EVAL_BNY, X_EVAL_STD, X_EVAL_LOG]
    Transform_list = ['None', 'PCA', 'LDA']
    Dims_list = [57, 15, 2]
    
    # RUN
    for idx, (X_train, X_eval, transform, dims) in enumerate(zip(X_train_list, X_eval_list, Transform_list, Dims_list)):
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("X_train = {}".format(X_train_list_names[idx]))
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        fit_SVM(X_train, T_TRAIN.astype(float), X_eval, T_EVAL.astype(float),
                kernel = gaussian_kernel, margin_type = 'soft', C=1.1, transform = transform, pp_dims = dims)