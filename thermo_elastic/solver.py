import torch as tc
import math

class fem_thermo_elastic_solver(tc.nn.Module):
    
    def __init__(self, options:dict, **kwargs):
        
        for _k, _v in options.items():
            
            setattr(self, _k, _v)
        
        # define initial distribution of nodes
        self.initial_mesh = tc.linspace(0.0, self.initial_length, self.number_nodes)
    
    def rpm(self, t, T):
        
        _u      = self._displacement_field(t, self.initial_mesh)
        _x_mesh = self.initial_mesh + _u
        _M      = self._assemble_mass_matrix(_x_mesh)
        _K      = self._assemble_stiffness_matrix(_x_mesh)
        _C      = self._assemble_advection_matrix(t, _x_mesh)
        _f      = self._assemble_forcing_matrix()
        _M_inv  = tc.linalg.pinv(_M)
        
        _T_dot = _M_inv @ (_f - (_K - _C) @ T)
        
        return _T_dot
        
    def _assemble_mass_matrix(self, x_mesh):
        
        _number_nodes = self.number_elements + 1
        _M            = tc.zeros((_number_nodes, _number_nodes))
        
        for e in range(self.number_elements):
            
            _i, _j = e, e+1
            dx_e   = x_mesh[_j] - x_mesh[_i]
            nodes  = [_i,_j]
            _Me    = (self.density * self.specific_heat * dx_e / 6.0) * tc.as_tensor([[2,1],[1,2]])
            
            for _i in range(2):
                
                for _j in range(2):
                    
                    _M[nodes[_i],nodes[_j]] += _Me[_i,_j]
                    
        return _M
        
    def _assemble_stiffness_matrix(self, x_mesh):
        
        _number_nodes = self.number_elements + 1
        _K            = tc.zeros((_number_nodes, _number_nodes))
        
        for e in range(self.number_elements):
            
            _i, _j = e, e+1
            dx_e   = x_mesh[_j] - x_mesh[_i]
            nodes  = [_i,_j]
            _Ke    = (self.thermal_conductivity / dx_e) * tc.as_tensor([[1,-1],[-1,1]])
            
            for _i in range(2):
                
                for _j in range(2):
                    
                    _K[nodes[_i],nodes[_j]] += _Ke[_i,_j]
            
        return _K
        
    def _assemble_forcing_matrix(self):
        
        _f      = tc.zeros((self.number_elements+1))
        _f[0]   = self.heat_flux
        
        return _f
    
    def _vm(self, t):
        
        _v_m = 1e-3
        
        return _v_m
    
    def _velocity_field(self, t, mesh):
        
        _vm = self._vm(t)
        
        return _vm * (1 - mesh / self.initial_length)
    
    def _displacement_field(self, t, mesh):
        
        _u_0 = self._vm(t) * (t - self.t_0)
        
        return _u_0 * (1.0 - mesh / self.initial_length)
    
    def _local_basis(self, xi):
        
        _phi   = tc.as_tensor([0.5 * (1 - xi), 0.5 * (1 + xi)])
        
        return _phi
    
    def _assemble_advection_matrix(self, t, x_mesh):
    
        C   = tc.zeros((self.number_nodes, self.number_nodes))
        
        for _e in range(self.number_elements):
            
            _i      = _e
            _j      = _e + 1
            x_e     = x_mesh[_i]
            x_ep1   = x_mesh[_j]
            dx_e    = x_ep1 - x_e
            dphi_dx = tc.as_tensor([-1 / dx_e, 1 / dx_e])
            x_c     = (x_ep1 + x_e) / 2
            J_e     = dx_e / 2
            
            # Gauss quadrature points
            xi_1, xi_2 = -1 / math.sqrt(3), 1 / math.sqrt(3)
            w_1, w_2   = 1.0, 1.0
            phi_1      = self._local_basis(xi_1)
            phi_2      = self._local_basis(xi_2)
            x_q1       = x_c + J_e * xi_1
            x_q2       = x_c + J_e * xi_2
            v_q1       = self._velocity_field(t, x_q1)
            v_q2       = self._velocity_field(t, x_q2)
            
            # Contribution from each quadrature point
            C1 = tc.outer(phi_1, dphi_dx) * w_1 * v_q1 * J_e
            C2 = tc.outer(phi_2, dphi_dx) * w_2 * v_q2 * J_e
            
            # Compute local matrix
            C_e = self.density * self.specific_heat * (C1 + C2)
            
            # Assemble global matrix
            
            C[_i:_i+2,_i:_i+2] += C_e
            
        return C
                
                
                
            
            
            
        
        