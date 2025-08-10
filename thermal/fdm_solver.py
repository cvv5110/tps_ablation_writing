import torch as tc

class fdm_thermal_solver(tc.nn.Module):
    
    def __init__(self, options: dict, **kwargs):
        
        for _k, _v in options.items():
            
            setattr(self, _k, _v)
            
        for _k, _v in kwargs.items():
            
            setattr(self, _k, _v)
            
        # define matrices
        self.A, self.b = self._assemble_matrices()
        self.x         = tc.linspace(0.0, self.length, self.number_nodes)
        
    def _assemble_matrices(self):
        
        _alpha = self.thermal_conductivity / (self.density * self.specific_heat)
        _r     = _alpha * self.dt / self.dx**2
        _A     = tc.zeros((self.number_nodes,self.number_nodes))
        _b     = tc.zeros((self.number_nodes))
        
        for _i in range(1, self.number_nodes - 1):
            
            _A[_i,_i-1] = _r
            _A[_i,_i]   = 1 - 2 * _r
            _A[_i,_i+1] = _r
        
        # left boundary (second order approximation with ghost node)    
        _A[0,0] = 1 - 2 * _r
        _A[0,1] = 2 * _r
        _b[0]   = 2 * self.dt * self.heat_flux / (self.density * self.specific_heat * self.dx)
        
        # right boundary (second order approximation)
        _A[-1,-2] = 2 * _r
        _A[-1,-1] = 1 - 2 * _r
        
        return _A, _b
    
    def rpm(self, t, T):
        
        _T_np1 = self.A @ T + self.b
        
        return _T_np1