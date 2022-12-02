import os
import gym
import numpy as np
from td3bc_smooth import Agent
import argparse
import rrc_2022_datasets

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--name", default="TD3BC_SM", type=str)
   parser.add_argument("--env", default="trifinger-cube-push-real-expert-v0", type=str)
   parser.add_argument("--sim_eval", default=True, type=bool)
   parser.add_argument("--sim_eval_freq", default=5000, type=int)
   parser.add_argument("--num_eval", default=10, type=int)
   parser.add_argument("--n_iteration", default=500000, type=int)
   parser.add_argument("--gamma", default=0.99, type=float)
   parser.add_argument("--tau", default=0.005, type=float)
   parser.add_argument("--batch_size", default=1024, type=int)
   parser.add_argument("--alpha", default=2.5, type=float)
   parser.add_argument("--lamd_s", default=0.1, type=float)
   parser.add_argument("--lr", default=0.0002, type=float)
   parser.add_argument("--pi_noise", default=0.2, type=float)
   parser.add_argument("--noise_clip", default=0.5, type=float)
   parser.add_argument("--pi_freq", default=2, type=int)
   parser.add_argument("--std_k", default=0.2, type=float)
   args = parser.parse_args()

   file_name = f"{args.name}_{args.env}"
   print("#######################################\n")
   print(f"Name: {args.name}\nEnv: {args.env}")
   print("\n#######################################")
   log_path = "./results/%s/%s"%(args.name, args.env)
   if not os.path.isdir("./results"): os.mkdir("./results")
   if not os.path.isdir("./results/%s" % args.name): os.mkdir("./results/%s" % args.name)
   if not os.path.isdir(log_path): os.mkdir(log_path)

   # Define environment and agent
   env = gym.make(
      args.env,
      disable_env_checker=True,
      visualization=False
   )
   action_max = env.action_space.high
   o_dim = env.observation_space.shape[0]
   a_dim = env.action_space.shape[0]
   agent = Agent(o_dim, a_dim, args)

   # Define evaluation environment (simulation is used)
   if args.sim_eval:
      eval_env = gym.make(
         args.env[:20] + 'sim-expert-v0',
         disable_env_checker=True,
         visualization=False
      )
   else:
      eval_env = None

   # Load the dataset (converted to transition samples by considering 'timeout' information.)
   dataset = env.get_dataset()
   o_mean = np.average(dataset["observations"], axis=0)
   o_std = np.std(dataset["observations"], axis=0)
   np.save(log_path + "/o_mean.npy", o_mean)
   np.save(log_path + "/o_std.npy", o_std)
   dataset["observations"] = (dataset["observations"] - o_mean) / (o_std + 1e-3)
   dataset["actions"] /= action_max
   od, nod, ad, rd = [], [], [], []
   ep_end_idx = np.hstack([np.array([-1]), np.where(dataset['timeouts'] == True)[0][:-1]])
   for k in range(len(ep_end_idx)):
      if k != len(ep_end_idx) - 1:
         od.append(dataset["observations"][ep_end_idx[k] + 1:ep_end_idx[k + 1]])
         nod.append(dataset["observations"][ep_end_idx[k] + 2:ep_end_idx[k + 1] + 1])
         ad.append(dataset['actions'][ep_end_idx[k] + 1:ep_end_idx[k + 1]])
         rd.append(dataset['rewards'][ep_end_idx[k] + 1:ep_end_idx[k + 1]].reshape([-1, 1]))
      else:
         od.append(dataset["observations"][ep_end_idx[k] + 1:-1])
         nod.append(dataset["observations"][ep_end_idx[k] + 2:])
         ad.append(dataset['actions'][ep_end_idx[k] + 1:-1])
         rd.append(dataset['rewards'][ep_end_idx[k] + 1:-1].reshape([-1, 1]))
   od = np.vstack(od).reshape([-1, o_dim]).astype(np.float32)
   nod = np.vstack(nod).reshape([-1, o_dim]).astype(np.float32)
   ad = np.vstack(ad).reshape([-1, a_dim]).astype(np.float32)
   rd = np.vstack(rd).reshape([-1, 1]).astype(np.float32)
   N = len(od)
   del dataset

   # Train
   global_step = 0
   sim_max_return = 0
   while global_step <= args.n_iteration:

      # Sample minibatch and train
      idx = np.random.choice(N, args.batch_size, replace=False)
      obs, next_obs, acts, rwds = od[idx], nod[idx], ad[idx], rd[idx]
      done = np.zeros_like(rwds).astype(np.float32)
      agent.train(obs, acts, rwds, next_obs, done)
      global_step += 1
      if global_step % 100 == 0:
         print(f"[#Training] Progress: {global_step} / {args.n_iteration}")

      # Simulation-based evaluation (just for reference)
      if global_step % args.sim_eval_freq == 0 and args.sim_eval:
         print("[#SimEval] Simulation-based evaluation...")
         rwd_set = []
         for _ in range(args.num_eval):
            obs = eval_env.reset()
            done = False
            total_reward = 0
            while not done:
               obs = (obs - o_mean) / (o_std + 1e-3)
               action = agent.get_action(obs)
               obs, rew, done, _ = eval_env.step(action * action_max)
               total_reward += rew
            rwd_set.append(total_reward)
         max_r = np.max(rwd_set)
         min_r = np.min(rwd_set)
         avg_r = np.mean(rwd_set)
         print(f"    - [Max]: {max_r} | [Min]: {min_r} | [Avg]: {avg_r}")
         if avg_r >= sim_max_return:
            agent.save_policy(log_path, post_fix="sim_best")
            sim_max_return = avg_r

   agent.save_policy(log_path)
   print("###############################################")
   print(f"[#INFO] ENVIRONMENT {args.env} - TRAINING IS DONE.")
   print("###############################################")
