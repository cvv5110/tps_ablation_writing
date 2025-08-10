import torch as tc

class fem_thermal_solver(tc.nn.Module):
    
    def __init__(self, options: dict, **kwargs):
        
        for _k, _v in options.items():
            
            setattr(self, _k, _v)
    
        # define system matrices
        
        self.M = self._assemble_mass_matrix()
        self.K = self._assemble_stiffness_matrix()
        self.f = self._assemble_forcing_matrix()
        self.x = tc.linspace(0.0, self.length, self.number_nodes)
        
    def rpm(self, t, T):
        
        _M_inv = tc.linalg.pinv(self.M)
        _T_dot = _M_inv @ (self.f - self.K @ T)
        
        return _T_dot
        
    def _assemble_mass_matrix(self):
        
        _number_nodes = self.number_elements + 1
        _M            = tc.zeros((_number_nodes, _number_nodes))
        _Me           = (self.density * self.specific_heat * self.dx / 6.0) * tc.as_tensor([[2,1],[1,2]])
        
        for e in range(self.number_elements):
            
            nodes = [e,e+1]
            
            for _i in range(2):
                
                for _j in range(2):
                    
                    _M[nodes[_i],nodes[_j]] += _Me[_i,_j]
                    
        return _M
        
    def _assemble_stiffness_matrix(self):
        
        _number_nodes = self.number_elements + 1
        _K            = tc.zeros((_number_nodes, _number_nodes))
        _Ke           = (self.thermal_conductivity / self.dx) * tc.as_tensor([[1,-1],[-1,1]])
        
        for e in range(self.number_elements):
            
            nodes = [e,e+1]
            
            for _i in range(2):
                
                for _j in range(2):
                    
                    _K[nodes[_i],nodes[_j]] += _Ke[_i,_j]
            
        return _K
        
    def _assemble_forcing_matrix(self):
        
        _f      = tc.zeros((self.number_elements+1))
        _f[0]   = self.heat_flux
        
        return _f
