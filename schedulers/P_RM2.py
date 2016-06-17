# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 06:41:25 2016

@author: Imane
"""

"""
Partitionned EDF using PartitionedScheduler.
"""
from simso.core.Scheduler import SchedulerInfo
from simso.utils import PartitionedScheduler
from simso.schedulers import scheduler

@scheduler("simso.schedulers.P_RM2")
class P_RM2(PartitionedScheduler):
    def init(self):
        PartitionedScheduler.init(
            self, SchedulerInfo("simso.schedulers.RM_mono"))

    def packer(self):
        # First Fit
        cpus = [cpu for cpu in self.processors]
        for task in self.task_list:
            for cpu in cpus:
				if cpu.identifier == task.data:
					# Affect it to the task.
					self.affect_task_to_processor(task, cpu)
        return True
