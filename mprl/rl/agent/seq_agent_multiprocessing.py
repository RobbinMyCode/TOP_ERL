import multiprocessing.connection
import multiprocessing
import torch

import mprl.rl.policy.abstract_policy as abs_policy
import mprl.rl.sampler.abstract_sampler as abs_sampler
import mprl.util as util
from mprl.rl.critic import SeqCritic
from mprl.rl.replay_buffer import SeqReplayBuffer

from .seq_agent import SeqQAgent


class SeqQAgentMultiProcessing(SeqQAgent):
    def __init__(self,
                 policy: abs_policy.AbstractGaussianPolicy,
                 critic: SeqCritic,
                 sampler: abs_sampler.AbstractSampler,
                 conn: multiprocessing.connection.Connection,
                 replay_buffer: SeqReplayBuffer,
                 projection=None,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 **kwargs):

        super().__init__(
            policy,
            critic,
            sampler,
            replay_buffer,
            projection,
            dtype,
            device,
            **kwargs,
        )
        self.conn = conn

    def step(self):
        # Update total step count
        self.num_iterations += 1

        # If logging data in the current step
        self.log_now = self.evaluation_interval == 1 or \
                       self.num_iterations % self.evaluation_interval == 1
        update_critic_now = self.num_iterations >= self.critic_update_from
        update_policy_now = self.num_iterations >= self.policy_update_from

        if not self.fresh_agent:
            buffer_is_ready = self.replay_buffer.is_full()
            self.num_iterations -= 1  # iteration only for collecting data
        else:
            buffer_is_ready = True

        # Note: Collect dataset until buffer size is greater than batch size
        while len(self.replay_buffer) < self.batch_size:
            self.conn.send((self.policy.parameters))
            dataset, num_env_interation = self.conn.recv()
            self.num_global_steps += num_env_interation
            dataset = self.process_dataset(dataset)

        util.run_time_test(lock=True, key="sampling")

        # NOTE: Update parameter of policy in the subprocess
        self.conn.send((self.policy.parameters))
        if update_critic_now and buffer_is_ready:
            # Update agent
            util.run_time_test(lock=True, key="update")

            critic_loss_dict, policy_loss_dict = self.update(update_policy_now)

            if update_critic_now and self.schedule_lr_critic:
                lr_schedulers = util.make_iterable(self.critic_lr_scheduler)
                for scheduler in lr_schedulers:
                    scheduler.step()

            if update_policy_now and self.schedule_lr_policy:
                self.policy_lr_scheduler.step()

            update_time = util.run_time_test(lock=False, key="update")

        else:
            critic_loss_dict, policy_loss_dict = {}, {}
            update_time = 0

        # NOTE: Wait for data from subprocess
        dataset, num_env_interation = self.conn.recv()

        self.num_global_steps += num_env_interation
        sampling_time = util.run_time_test(lock=False, key="sampling")

        # Process dataset and save to RB
        util.run_time_test(lock=True, key="process_dataset")
        dataset = self.process_dataset(dataset)
        process_dataset_time = util.run_time_test(lock=False,
                                                key="process_dataset")

        # Log data
        if self.log_now and buffer_is_ready:
            # Generate statistics for environment rollouts
            dataset_stats = \
                util.generate_many_stats(dataset, "exploration", to_np=True,
                                         exception_keys=["decision_idx"])

            # Prepare result metrics
            result_metrics = {
                **dataset_stats,
                "sampling_time": sampling_time,
                "process_dataset_time": process_dataset_time,
                "num_global_steps": self.num_global_steps,
                **critic_loss_dict, **policy_loss_dict,
                "update_time": update_time,
                "lr_policy": self.policy_lr_scheduler.get_last_lr()[0]
                if self.schedule_lr_policy else self.lr_policy,
                "lr_critic1": self.critic_lr_scheduler[0].get_last_lr()[0]
                if self.schedule_lr_critic else self.lr_critic,
                "lr_critic2": self.critic_lr_scheduler[1].get_last_lr()[0]
                if self.schedule_lr_critic else self.lr_critic
            }

            # Evaluate agent
            util.run_time_test(lock=True)
            evaluate_metrics = util.generate_many_stats(
                self.evaluate()[0], "evaluation", to_np=True,
                exception_keys=["decision_idx"])
            evaluation_time = util.run_time_test(lock=False)
            result_metrics.update(evaluate_metrics),
            result_metrics.update({"evaluation_time": evaluation_time}),
        else:
            result_metrics = {}

        return result_metrics