class ParametricRBFMapping(BaseParametric):
    """
    Parametric mapping class for RBF parameterization.
    This class is used to transform model parameters using radial basis functions (RBFs).
    It is designed to work with PyTorch for automatic differentiation.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The mesh on which the model is defined.
    nP : int
        Number of parameters.
    active_cells : numpy.ndarray,
        Indices of active cells.
    forward_transform : callable
        Function to perform the forward transformation.
    inverse_transform : callable
        Function to perform the inverse transformation.
    params : array_like
        Additional parameters for the transformation.
    deriv_calc : str
        Type of derivative calculation ('user' or 'auto').

    """

    def __init__(self, mesh=None, nP=None, active_cells=None, forward_transform=None, inverse_transform=None,
                 params=None, deriv_calc='user', **kwargs):
        self.mesh = mesh
        self._nP = nP
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.active_cells = active_cells
        self.params = params
        self.deriv_calc = deriv_calc
        if self.z is not None:
            self.xyz = np.vstack((self.x, self.y, self.z)).T
        else:
            self.xyz = np.vstack((self.x, self.y)).T


    def _transform(self, m):
        '''
        model parameters
        '''
        param_model = m
        if self.params is not None:
            return self.forward_transform(self.params, self.xyz, param_model)
        else:
            return self.forward_transform(param_model)

    def deriv(self, m, v=None):
        deriv_calc = self.deriv_calc
        if deriv_calc == 'user':
           param_model = (m)
           if v is not None:
               v_loc = (v)
               return sp.csr_matrix(self.forward_transform(self.params, self.xyz, param_model, return_deriv=True)) * v
           else:
               return sp.csr_matrix(self.forward_transform(self.params, self.xyz, param_model, return_deriv=True))#.detach().numpy()

        elif deriv_calc == 'auto':
            # param_model = torch.tensor(m, dtype=torch.float64 ,requires_grad=True)
            # if v is not None:
            #     v_loc = torch.tensor(v)
            #     temp = jvp(lambda x: self.forward_transform(self.params,self.xyz, x), param_model, v_loc)
            #     return sp.csr_matrix(
            #         temp[1].numpy())
            # else:
            #     temp = jacobian(lambda x: self.forward_transform(self.params,self.xyz, x), param_model, 
            #                     strategy='reverse-mode',vectorize=True, create_graph=False)
            #     return sp.csr_matrix(temp
            #                          .detach().numpy())
            raise NotImplementedError("Automatic differentiation is not yet implemented for this class.")
    @property
    def nP(self):
       return self._nP

    @property
    def shape(self):
        if self.active_cells is not None:
               return (self.active_cells.sum(), self._nP)
        else:
           return (self.mesh.n_cells, self._nP)