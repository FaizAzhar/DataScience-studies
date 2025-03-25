import numpy as np
import autodiff as AD

EPS = 1e-5

class MathDumpster():
    def __init__(self):
        pass
    
    def Hessian(self, par, fn):
        """
        Compute the Hessian of a function at a given point
        using Automatic Differentiation.
        Args:
            par (array-like): The point at which to compute the hessian.
            fn (callable): The function to differentiate.
        Returns:
            np.array: Hessian matrix.
        """
        par = [float(i) for i in par]
        npar = len(par)
        hess = np.zeros((npar,npar))
        par = [AD.Var('',i) for i in par]
        
        for i in range(len(par)):
            for j in range(len(par)):
                hess[i][j] = fn(par).backward(par[i]).backward(par[j]).compute()
        return(hess)
    
    def Differentiate(self, par, fn):
        """
        Compute the gradient (partial derivatives) of a function at a given point
        using finite differences.
        Args:
            par (array-like): The point at which to compute the gradient.
            fn (callable): The function to differentiate.
        Returns:
            np.array: Gradient vector.
        """
        par = [float(i) for i in par]
        par = np.array(par)
        grad = np.zeros_like(par)
        
        for i in range(len(par)):
            par_forward = par.copy()
            par_backward = par.copy()
            par_forward[i] += EPS  # Increment one element
            par_backward[i] -= EPS  # Decrement one element
            grad[i] = (fn(par_forward) - fn(par_backward)) / (2*EPS)
        return(grad)
    
    def AutoDiff(self, par, fn):
        """
        Compute the gradient (partial derivatives) of a function at a given point
        using Automatic Differentiation.
        Args:
            par (array-like): The point at which to compute the gradient.
            fn (callable): The function to differentiate.
        Returns:
            np.array: Gradient vector.
        """
        par = [float(i) for i in par]
        grad = np.zeros_like(par)
        par = [AD.Var('',i) for i in par]
        
        for i in range(len(par)):
            grad[i] = fn(par).backward(par[i]).compute()
        return(grad)
    
    def sin(self, x):
        if not(isinstance(x, AD.DifferentiableSymbolicOperation)):
            x = AD.Var('',x)
        return AD.Sin(x)
    
    def cos(self, x):
        if not(isinstance(x, AD.DifferentiableSymbolicOperation)):
            x = AD.Var('',x)
        return AD.Cos(x)
    
    def tan(self, x):
        if not(isinstance(x, AD.DifferentiableSymbolicOperation)):
            x = AD.Var('',x)
        return AD.Tan(x)
    
    def exp(self, x):
        if not(isinstance(x, AD.DifferentiableSymbolicOperation)):
            x = AD.Var('',x)
        return AD.Exp(x)
    
    def log(self, x):
        if not(isinstance(x, AD.DifferentiableSymbolicOperation)):
            x = AD.Var('',x)
        return AD.Log(x)