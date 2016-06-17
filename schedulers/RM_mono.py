"""
Rate Monotic algorithm for uniprocessor architectures.
"""
from simso.core import Scheduler
from simso.schedulers import scheduler

@scheduler("simso.schedulers.RM_mono")
class RM_mono(Scheduler):
    def init(self):
        self.ready_list = []

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()
        print job.cpu.name

    def on_terminated(self, job):
        self.ready_list.remove(job)
        job.cpu.resched()

    def schedule(self, cpu):
        if self.ready_list:
            # job with the highest priority
            job = min(self.ready_list, key=lambda x: x.period)
        else:
            job = None

        return (job, cpu)
    def packer(self):
        # First Fit
        cpus = [cpu for cpu in self.processors]
        for task in self.task_list:
            for cpu in cpus:
				if cpu.identifier == task.data:
					# Affect it to the task.
					self.affect_task_to_processor(task, cpu)
        return True