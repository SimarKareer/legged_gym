from legged_gym.lbc.algorithms.lbc import LBC
from legged_gym.lbc.models.actor_encoder_lbc import Actor, VisionEncoder
from rsl_rl.modules.actor_critic import DmEncoder
from torch.utils.tensorboard import SummaryWriter
import torch
from collections import deque
import statistics
import os
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from pytorch_memlab import MemReporter
import time



class LbcRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.lbc_cfg = train_cfg["lbc"]
        self.device = device
        self.env = env
        envcfg = self.env.cfg

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic


        print("traincfg: ", train_cfg)
        # enc_inp_size = train_cfg["obsSize"]["encoder_input_size"]
        # enc_out_size = train_cfg["obsSize"]["encoder_output_size"]
        enc_hidden_dims = train_cfg["obsSize"]["encoder_hidden_dims"]
        num_dm_encoder_obs = train_cfg["obsSize"]["num_dm_encoder_obs"]


        # num_encoder_obs = self.env.num_obs - envcfg.env.num_proprio_obs
        num_actor_obs = envcfg.env.num_proprio_obs + train_cfg["obsSize"]["cnn_out_size"]

        
        # actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor = Actor(
            num_actor_obs, self.env.num_actions
        ).to(self.device)
        actorDict = torch.load(train_cfg["runner"]["teacher_policy"])["model_state_dict"]
        actor.load_state_dict(actorDict, strict=False)

        vision_encoder = VisionEncoder()
        dm_encoder = DmEncoder(num_dm_encoder_obs, enc_hidden_dims)
        dmDict = torch.load(train_cfg["runner"]["teacher_policy"])["model_state_dict"]
        newDmDict = {}
        for k, v in dmDict.items():
            if "encoder" in k:
                newk = k[8:]
                newDmDict[newk] = v
        # print("dm dict: ", newDmDict.keys())
        dm_encoder.load_state_dict(newDmDict)

        
        # alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg = LBC(actor, vision_encoder, dm_encoder, device="cuda", learning_rate=1e-4)
        # self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        # self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        # self.alg.init_storage(
        #     self.env.num_envs,
        #     self.num_steps_per_env,
        #     [self.env.num_obs],
        #     [self.env.num_privileged_obs],
        #     [self.env.num_actions],
        # )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    # def learn2(self):

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        # privileged_obs = self.env.get_privileged_observations()
        # critic_obs = privileged_obs if privileged_obs is not None else obs
        # obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        obs = obs.to(self.device)
        self.alg.train_mode()

        # print(self.lbc_cfg)
        # print(self.lbc_cfg["lbc"])
        self.num_steps_per_env = self.lbc_cfg["batch_size"]


        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        reporter = MemReporter()

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # reporter.report()
            # Rollout
            # with torch.inference_mode():
            batch_loss = 0
            for i in range(self.num_steps_per_env):
                # enc_dm = self.alg.encode_dm(obs)
                # enc_visual = self.alg.encode_visual(obs)

                actions, loss = self.alg.act_and_sl(obs)
                # print("loss: ", loss)
                
                print("calling step")
                (
                    obs,
                    privileged_obs,
                    rewards,
                    dones,
                    infos,
                ) = self.env.step(actions)

                self.alg.process_env_step(rewards, dones, infos)
                print("stumble: ", self.env.extras["episode"]["eval_feet_step"])

                # self.alg.update(loss)

                # print("Updating")
                batch_loss += loss

                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            batch_loss /= self.num_steps_per_env
            # print("loss: ", batch_loss)
            self.alg.update(batch_loss)

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            # self.alg.compute_returns(critic_obs)


            # mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            self.log(locals())
            
            batch_loss = 0
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        # mean_std = self.alg.actor_critic.std.mean()
        # mean_std = None
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/lbc_loss", locs["batch_loss"], locs["it"]
        )

        # self.writer.add_scalar(
        #     "Loss/value_function", locs["mean_value_loss"], locs["it"]
        # )
        # self.writer.add_scalar(
        #     "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        # )
        # self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        # self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                # f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                # f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                # f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'LBC Loss:':>{pad}} {locs['batch_loss']:.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                # f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                # f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                # f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'LBC Loss:':>{pad}} {locs['batch_loss']:.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "actor_state_dict": self.alg.actor.state_dict(),
                "vision_encoder_state_dict": self.alg.vision_encoder.state_dict(),
                "dm_encoder_state_dict": self.alg.dm_encoder.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
        self.alg.vision_encoder.load_state_dict(loaded_dict["vision_encoder_state_dict"])
        self.alg.dm_encoder.load_state_dict(loaded_dict["dm_encoder_state_dict"])
        
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        # self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        # if device is not None:
        #     self.alg.actor_critic.to(device)
        # return self.alg.actor_critic.act_inference

        self.alg.actor.eval()
        self.alg.vision_encoder.eval()
        return self.alg.act_inference
