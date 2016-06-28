from .WCET import WCET
from .ACET import ACET
from .CacheModel import CacheModel
from .RandomCache import RandomCache
from .FixedPenalty import FixedPenalty

execution_time_models = {
    'wcet': WCET,
    'acet': ACET,
    'cache': CacheModel,
    'fixedpenalty': FixedPenalty,
    'test' : RandomCache
}

execution_time_model_names = {
    'WCET': 'wcet',
    'ACET': 'acet',
    'Cache Model': 'cache',
    'Fixed Penalty': 'fixedpenalty',
    'Random Cache' :'test'
}
