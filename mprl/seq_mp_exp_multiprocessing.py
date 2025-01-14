import multiprocessing.connection
import os
from torch import multiprocessing as mp
import multiprocessing
from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from tqdm import tqdm

import mprl.util as util
from mprl.rl.agent import agent_factory
from mprl.rl.critic import critic_factory
from mprl.rl.policy import policy_factory
from mprl.rl.projection import projection_factory
from mprl.rl.sampler import sampler_factory
from mprl.rl.replay_buffer import replay_buffer_factory
import psutil
import copy
import time

# NOTE: task in the sub process
def sampler_task(cfg, cpu_cores, conn: multiprocessing.connection.Connection):
    sampler = sampler_factory(
        cfg["sampler"]["type"],
        cpu_cores=cpu_cores,
        disable_test_env=True,
        **cfg["sampler"]["args"])

    state_dim = MPExperimentMultiProcessing.get_dim_in(cfg, sampler)
    policy_out_dim = MPExperimentMultiProcessing.dim_policy_out(cfg)

    cfg_copy = copy.deepcopy(cfg) #contextual gets deleted somehow

    inference_policy = policy_factory(
        cfg["policy"]["type"],
        dim_in=state_dim,
        dim_out=policy_out_dim,
        **cfg["policy"]["args"]
    )

    n_corrections = len(cfg_copy["sampler"]["args"].get("correction_steps", []))
    #if n_corrections >= 1:
    secondary_training = n_corrections >= 1
    secondary_steps = cfg_copy["sampler"]["args"]["correction_steps"] if secondary_training else None
    secondary_policies = len(secondary_steps) * [
        policy_factory(
            cfg_copy["policy"]["type"],
            dim_in=state_dim,
            dim_out=policy_out_dim,
            **cfg_copy["policy"]["args"]
        )
    ] if secondary_training else None


    # NOTE: run the sampler and send the data to the main process
    # NOTE: It is while True so I am not sure if it is proper to use it
    while True:
        # NOTE: get the parameters of the inference policy and critic from the main process
        # NOTE: recv() is a blocking function, it will be waiting until the main process sends the parameters
        policy_params = conn.recv()
        # NOTE: you can send a False to stop the process, like this:
        # NOTE: if policy_params is False:
        # NOTE:     break
        # TODO: ALLOW SKIP THE PARAMETERS COPY USING THE RECIEVED DATA
        inference_policy.copy_parameter(policy_params)
        dataset, num_env_interation = sampler.run(
            training=True, policy=inference_policy, critic=None, secondary_training=secondary_training,
            secondary_policies=secondary_policies, secondary_steps=secondary_steps)
        # Send the data to the main process
        conn.send((dataset, num_env_interation))

class MPExperimentMultiProcessing(experiment.AbstractIterativeExperiment):
    def initialize(self, cw_config: dict, rep: int,
                   logger: cw_logging.LoggerArray) -> None:
        #print("cw_config:", cw_config)
        #exit("main_74")
        # Get experiment config
        cfg = cw_config["params"]
        cpu_cores = cw_config.get("cpu_cores", None)
        if cpu_cores is None:
            cpu_cores = set(range(psutil.cpu_count(logical=True)))
        # Set random seed globally
        util.set_global_random_seed(cw_config["seed"])
        self.verbose_level = cw_config.get("verbose_level", 1)

        # Determine training or testing mode
        load_model_dir = cw_config.get('load_model_dir', None)
        load_model_epoch = cw_config.get('load_model_epoch', None)

        if load_model_dir is None or cw_config["keep_training"]:
            self.training = True
        else:
            self.training = False

        if self.training and cw_config.get("save_model_dir", None) is not None:
            # Save model in training mode
            self.save_model_dir = os.path.abspath(cw_config["save_model_dir"])
            self.save_model_interval = \
                max(cw_config["iterations"] // cw_config["num_checkpoints"], 1)

        else:
            # In testing mode or no save model dir in training mode
            self.save_model_dir = None
            self.save_model_interval = None

        # NOTE: I have no idea what happens here
        # NOTE: I must deepcopy the config to avoid the error
        # NOTE: It seems that some code change the cfg so that sub process can not use it
        cfg_copy = copy.deepcopy(cfg)

        # Components
        self.sampler = sampler_factory(cfg["sampler"]["type"],
                                       cpu_cores=cpu_cores,
                                       disable_train_env=True,
                                       **cfg["sampler"]["args"])

        state_dim = self.get_dim_in(cfg, self.sampler)
        policy_out_dim = self.dim_policy_out(cfg)

        self.policy = policy_factory(cfg["policy"]["type"],
                                     dim_in=state_dim,
                                     dim_out=policy_out_dim,
                                     **cfg["policy"]["args"])
        action_dim = self.policy.num_dof * 2

        self.critic = critic_factory(cfg["critic"]["type"],
                                     state_dim=state_dim,
                                     action_dim=action_dim,
                                     **cfg["critic"]["args"])
        self.projection = projection_factory(
            cfg["projection"]["type"],
            action_dim=self.dim_policy_out(cfg),
            **cfg["projection"]["args"]
        )
        # NOTE: pipe for exchanging the data and parameters between the main process and the sub process
        self.main_conn, self.sub_conn = mp.Pipe()
        # NOTE: CUDA needs a spawn context for multiprocessing
        ctx = mp.get_context('spawn')
        self.sampler_process = ctx.Process(
            target=sampler_task,
            args=(cfg_copy, cpu_cores, self.sub_conn)
        )
        self.sampler_process.start()
        util.assign_process_to_cpu(self.sampler_process.pid, cpu_cores)
        traj_length = self.sampler.num_times

        replay_buffer_data_shape = {
            "step_states": (traj_length, state_dim),
            "step_desired_pos": (traj_length, self.policy.num_dof),
            "step_desired_vel": (traj_length, self.policy.num_dof),
            "step_actions": (traj_length, action_dim),
            "step_rewards": (traj_length,),
            "step_dones": (traj_length,),
            "decision_idx": (),
            "segment_init_time": (),
            "segment_init_pos": (self.policy.num_dof,),
            "segment_init_vel": (self.policy.num_dof,),
            "segment_params_mean": (policy_out_dim,),
            "segment_params_L": (policy_out_dim, policy_out_dim),
            "segment_reward": (),
        }

        if cfg["agent"]["args"].get("norm_obs", False):
            replay_buffer_norm_info = {
                "step_states": True,
                "step_desired_pos": False,
                "step_desired_vel": False,
                "step_actions": True,
                "step_rewards": False,
                "step_dones": False,
                "decision_idx": False,
                "segment_init_time": False,
                "segment_init_pos": False,
                "segment_init_vel": False,
                "segment_params_mean": False,
                "segment_params_L": False,
                "segment_reward": False,
            }
        else:
            replay_buffer_norm_info = None

        self.replay_buffer = replay_buffer_factory(
            cfg["replay_buffer"]["type"],
            data_info=replay_buffer_data_shape,
            data_norm_info=replay_buffer_norm_info,
            **cfg["replay_buffer"]["args"])

        self.agent = agent_factory(cfg["agent"]["type"],
                                   policy=self.policy,
                                   critic=self.critic,
                                   sampler=self.sampler,
                                   conn=self.main_conn,
                                   projection=self.projection,
                                   replay_buffer=self.replay_buffer,
                                   traj_length=traj_length,
                                   **cfg["agent"]["args"])

        # Load model if it in testing mode
        if load_model_dir is None:
            util.print_line_title("Training")
        else:
            self.agent.load_agent(load_model_dir, load_model_epoch)
            util.print_line_title("Testing")

        # Progressbar
        self.progress_bar = tqdm(total=cw_config["iterations"])

        # Running speed log
        self.exp_start_time = time.perf_counter()

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        if self.training:
            result_metrics = self.agent.step()
            result_metrics["exp_speed"] = self.experiment_speed(n)
            # print(f"Speed: {result_metrics['exp_speed']:.2f} s/iter")

            self.progress_bar.update(1)
            #(result_metrics)
            #exit("main201")
            if self.verbose_level == 0:
                return {}
            elif self.verbose_level == 1:
                # Delete unnecessary logging data for lowering wandb bandwidth
                # Very important for metaworld experiments with all seeds
                for key in dict(result_metrics).keys():
                    if ("exploration" in key
                            or "projection" in key
                            or "gradient" in key
                            or "grad_norm" in key
                            or "clipped_" in key
                            or "mc_returns" in key
                            or "targets_bias" in key
                            or "step_actions" in key
                            or "step_states" in key
                            or "step_rewards" in key
                            or "step_desired_pos" in key
                            or "step_desired_vel" in key
                            or "step_dones" in key
                            or "time_limit_dones" in key
                            or "segment" in key
                            or "update_" in key
                            or "median" in key
                            or "targets" in key
                            or "entropy" in key
                            or "trust_region" in key
                            or "loss" in key
                            or "critic" in key
                    ):
                        del result_metrics[key]
                return result_metrics
            elif self.verbose_level == 2:
                return result_metrics

        else:
            # Note: Use the below line to train a loaded model for long term bug
            # self.agent.step()

            deterministic_result_dict, _ = self.agent.evaluate(render=True)
            self.progress_bar.update(1)
            return deterministic_result_dict

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        if self.save_model_dir and ((n + 1) % self.save_model_interval == 0
                                    or (n + 1) == cw_config["iterations"]):
            self.agent.save_agent(log_dir=self.save_model_dir, epoch=n + 1)

    def finalize(self,
                 surrender: cw_error.ExperimentSurrender = None,
                 crash: bool = False):
        self.sampler_process.terminate()

    @staticmethod
    def get_dim_in(cfg, sampler):
        """
        Get the dimension of the policy and critic input

        Args:
            cfg: config dict
            sampler: sampler of the experiment

        Returns:
            dim_in: dimension of the policy output

        """
        if "TemporalCorrelated" in cfg["sampler"]["type"] \
                or "SeqSampler" in cfg["sampler"]["type"]:
            dof = cfg["mp"]["args"]["num_dof"]
            return sampler.observation_shape[-1] - dof * 2
        else:
            return sampler.observation_shape[-1]

    @staticmethod
    def dim_policy_out(cfg):
        """
        Get the dimension of the policy output

        Args:
            cfg: config dict

        Returns:
            dim_out: dimension of the policy output

        """
        mp_type = cfg["mp"]["type"]
        dof = cfg["mp"]["args"]["num_dof"]
        num_basis = cfg["mp"]["args"]["num_basis"]
        learn_tau = cfg["mp"]["args"].get("learn_tau", False)
        learn_delay = cfg["mp"]["args"].get("learn_delay", False)

        if mp_type == "prodmp":
            dim_out = dof * (num_basis + 1)  # weights + goal

            # Disable goal if specified
            if cfg["mp"]["args"].get("disable_goal", False):
                dim_out -= dof

        elif mp_type == "promp":
            dim_out = dof * num_basis  # weights only
        else:
            raise NotImplementedError

        if learn_tau:
            dim_out += 1
        if learn_delay:
            dim_out += 1

        return dim_out

    def experiment_speed(self, n):
        current_time = time.perf_counter()
        # Time in seconds per iteration
        return (current_time - self.exp_start_time) / (n + 1)


def evaluation(model_str: str, version_number: list, epoch: int,
               keep_training: bool, experiment=MPExperimentMultiProcessing):
    """
    Given wandb model string, version, and epoch number, evaluate the model
    Args:
        model_str: wandb model string
        version_number: number of the version
        epoch: epoch number of the model
        keep_training: whether to keep training the model

    Returns:
        None
    """
    for v_num in version_number:
        util.RLExperiment(experiment, False, model_str, v_num, epoch,
                          keep_training)



if __name__ == "__main__":
    # import numpy as np
    # np.seterr(all='raise')
    for key in os.environ.keys():
        if "-xCORE-AVX2" in os.environ[key]:
            os.environ[key] = os.environ[key].replace("-xCORE-AVX2", "")

    util.RLExperiment(MPExperimentMultiProcessing, True)
