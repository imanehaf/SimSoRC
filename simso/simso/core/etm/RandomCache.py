# coding=utf-8

from simso.core.etm.AbstractExecutionTimeModel \
    import AbstractExecutionTimeModel

# polynomial
from numpy.polynomial import Polynomial as Poly


def getReuse(ins):
    """
    Get reuse distance for given addresses.
    """
    # reuse distance
    reuse = []
    # previous instruction
    prevIns = []
    # previous instruction index number
    prevNum = []

    # prob. for instuctions
    for i in range(len(ins)):
        w = ins[i]
        if w in prevIns:
            K = i-prevNum[prevIns.index(w)]-1
            reuse.append(K)
            prevNum[prevIns.index(w)] = i
        else: # first time use
            prevIns.append(w)
            prevNum.append(i)
            reuse.append('inf')

    return reuse

def getHit(reuse, N):
    """
    Get hit probabilities using reuse distance and associativity.
    """
    Phit = []
    for K in reuse.keys():
        if K<N:
            P = (1-1.0/N)**K
			for i in range(reuse[K]):
				Phit.append(P)
        else:
			for i in range(reuse[K]):
				Phit.append(0)
    return Phit

def add_time_prob(time, cycles, prob, cyclesP):
    """
    Convolution of two Execution Time Profiles.
    """
    p1 = [0]*(max(time)+1)
    for i in range(len(time)):
        p1[time[i]] = prob[i]

    p2 = [0]*(max(cycles)+1)
    for i in range(len(cycles)):
        p2[cycles[i]] = cyclesP[i]

    # polynomial mult.
    p1 = Poly(p1)
    p2 = Poly(p2)
    p = p1*p2
    p = p.coef.tolist()

    # get cycles and prob.
    time = []
    prob = []
    for i in range(len(p)):
        #if p[i]>0:
        time.append(i)
        prob.append(p[i])

    return time, prob

def calc_cpi(hit_rates, latencies):
    cpi = 0
    
    for i in range(len(latencies)):
        cpi += hit_rates[i] * latencies[i]

    return cpi

def capacity_miss_random(rd, cache_size):
    """
    Obtain local miss ratio.
    """
    N = cache_size

    Phit = getHit(rd, N)
 
    # miss counts and prob.
    time = [0]
    prob = [1.0]
    for j in range(len(Phit)):
        Nc = [0, 1]
        Pdf = [Phit[j], 1-Phit[j]]
        time, prob = add_time_prob(time, Nc, prob, Pdf)

    # local miss ratio
    mr = 0
    for j in range(len(time)):
        mr += time[j]*prob[j]

    mr /= time[-1]
    
    return mr


def cpi_alone(task, cache_sizes, latencies):
    local_miss_rates = [capacity_miss_random(task.prd, cache_size)
                  for cache_size in cache_sizes]

    hit_rates = [0]*len(local_miss_rates)
    # previous miss ratio product
    mrp = 1.0
    for i, d in enumerate(local_miss_rates):
        hit_rates[i] = mrp * (1 - d)
        mrp *= d

    return calc_cpi(hit_rates, latencies)


def calc_prds(caches, task, running_jobs):
    """
    Compute the interleaving reuse distance profiles.
    """
    result = {}
    for cache in caches:
        shared_jobs = [j for j in running_jobs if j.cpu in cache.shared_with]

        cpi = task.get_cpi_alone()
        sum_cpi = sum(cpi / j.task.get_cpi_alone() for j in shared_jobs)
        rd, freq = task.mt.prd().items()#TODO
		for i in range(freq):
			tmp = [r*sum_cpi for r in rd]
		result[tmp] = freq
    return result


def compute_instructions(task, running_jobs, duration):
    caches = task.cpu.caches
    latencies = [c.access_time for c in caches]
    prds = calc_prds(caches, task, running_jobs) 
    hit_rates = [capacity_miss_random(prd, cache.size) for (prd, cache) in zip(prds, caches)]
    return duration / calc_cpi(hit_rates, latencies)

def penalty_to_latency(penalty_memaccess, penalties):
    latencies = penalties
    for i in range(1, len(latencies)):
        latencies[i] += latencies[i-1]                
    latencies = [i+penalty_memaccess for i in latencies]

    return latencies
   

class RancomCache(AbstractExecutionTimeModel):
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

        # precompute cpi_alone for each task on each cpu
        for task in self.sim.task_list:
            for proc in self.sim.processors:
                caches = proc.caches
                
                task.set_cpi_alone(
                    proc,
                    cpi_alone(task, [c.size for c in caches], 
                                [c.access_time for c in caches])
                )

    def update(self):
        self._update_instructions()

    def _update_instructions(self):
        for job in self._running_jobs:
            # Compute number of instr for self.sim.now() - last_update
            instr = compute_instructions(job.task, self._running_jobs,
                                         self.sim.now() - self._last_update)
            # Update the number of instr for this job
            self._instr_jobs[job] = self._instr_jobs.get(job, 0) + instr

        # Update last_update
        self._last_update = self.sim.now()

    def on_activate(self, job):
        self.penalty[job] = 0

    def on_execute(self, job):
        # Compute penalty.
        if job in self.was_running_on:
            # resume on the same processor.
            if self.was_running_on[job] is job.cpu:
                if self.running[job.cpu] is not job:
                    self.penalty[job] += job.task.preemption_cost
            else:  # migration.
                self.penalty[job] += job.task.preemption_cost

        self.running[job.cpu] = job
        self.was_running_on[job] = job.cpu

        # Update the number of instructions executed for the running jobs.
        self._update_instructions()
        # Add the job in the list of running jobs.
        self._running_jobs.add(job)

    def _stop_job(self, job):
        # Update the number of instructions executed for the running jobs.
        self._update_instructions()
        # Remove the job from the list of running jobs.
        self._running_jobs.remove(job)

    def on_preempted(self, job):
        self._stop_job(job)

    def on_terminated(self, job):
        self._stop_job(job)

    def on_abort(self, job):
        self._stop_job(job)

    def get_ret(self, job):
        self._update_instructions()
        penalty = self.penalty[job]
        return (job.task.n_instr - self._instr_jobs[job]) \
            * job.task.get_cpi_alone() + penalty
