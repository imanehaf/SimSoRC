# coding=utf-8

from simso.core.etm.AbstractExecutionTimeModel \
    import AbstractExecutionTimeModel




class test_etm(AbstractExecutionTimeModel):
    def __init__(self, sim, nb_processors):
        self.sim = sim
        self._nb_processors = nb_processors

    def init(self):
        self._last_update = 0
        self._running_jobs = set()
        self._instr_jobs = {}
        self._total_preemptions_cost = 0
        self.running = {}
        self.penalty = {}
        self.was_running_on = {}


    def update(self):

    

    def on_activate(self, job):

    def on_execute(self, job):
       

    def on_preempted(self, job):


    def on_terminated(self, job):

    def on_abort(self, job):

    def get_ret(self, job):
