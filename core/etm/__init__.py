from .WCET import WCET
from .ACET import ACET
from .CacheModel import CacheModel
from .FixedPenalty import FixedPenalty
from .test_etm import test_etm

execution_time_models = {
    'wcet': WCET,
    'acet': ACET,
    'cache': CacheModel,
    'fixedpenalty': FixedPenalty,
	'test' : test_etm
}

execution_time_model_names = {
    'WCET': 'wcet',
    'ACET': 'acet',
    'Cache Model': 'cache',
    'Fixed Penalty': 'fixedpenalty',
	'Test' :'test'
}
