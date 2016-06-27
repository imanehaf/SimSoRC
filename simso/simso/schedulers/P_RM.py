"""
Partitionned EDF using PartitionedScheduler.
"""
from simso.core.Scheduler import SchedulerInfo
from simso.utils import PartitionedScheduler
from simso.schedulers import scheduler

@scheduler("simso.schedulers.P_RM")
class P_RM(PartitionedScheduler):
    def init(self):
        self.ready_list = []
        self.map_cpu_sched = {}
        # Mapping task to scheduler.
        self.map_task_sched = {}

        for cpu in self.processors:
            # Instantiate a scheduler.
            scheduler_info = SchedulerInfo("simso.schedulers.RM_mono")
            sched = scheduler_info.instantiate(self.sim)
            sched.add_processor(cpu)

            # Affect the scheduler to the processor.
            self.map_cpu_sched[cpu.identifier] = sched

        self._packer = self.packer
        assert self.packer(), "Packing failed"

        for cpu in self.processors:
            self.map_cpu_sched[cpu.identifier].init()

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()
        #print job.cpu.name

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
        cpus = [cpu for cpu in self.processors]
        for task in self.task_list:
            for cpu in cpus:
                if cpu.identifier == task.data:
                    self.affect_task_to_processor(task, cpu)
                    print ("affecting %s to %s\n"%(task.name, cpu.name))
        return True
