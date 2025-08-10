opts = {
    'problem_name'         : '1d_thermal',
    'density'              : 4_500,
    'thermal_conductivity' : 21.9,
    'specific_heat'        : 522,
    'length'               : 0.02,
    'initial_temperature'  : 300.0,
    'number_elements'      : 50,
    'number_nodes'         : 51,
    'dx'                   : 0.02 / 50,
    'number_time_points'   : 100,
    'heat_flux'            : 1e5,
    'dt'                   : 0.001,
}