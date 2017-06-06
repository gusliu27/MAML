# Python Imports
import argparse
import numpy as np
# Torch Imports
import torch
from models import *
#from torch import LongTensor
#from torchtext.vocab import load_word_vectors
from torch.autograd import Variable
#from torch.nn.utils.rnn import pack_padded_sequence#, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch import cuda, FloatTensor
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import copy
import gym

class Config:
  def __init__(self, args):
    self.num_rollouts = args.nr
    self.gamma = args.g
    self.h = args.h
    self.alpha = args.a
    self.beta = args.b
    self.hidden_size = args.hs
    self.meta_batch_size = args.bs
    self.num_meta_iterations = args.nmi
    #self.eval_every = args.ee
    self.use_gpu = args.gpu
    self.dtype = FloatTensor
    self.goal_range = args.gr
    self.action_space_size = args.asp
    self.state_space_size = args.ssp
    self.reward_type = args.rew


  def __str__(self):
    properties = vars(self)
    properties = ["{} : {}".format(k, str(v)) for k, v in properties.items()]
    properties = '\n'.join(properties)
    properties = "--- Config --- \n" + properties + "\n"
    return properties


def parseConfig(description="Default Model Description"):
  parser = argparse.ArgumentParser(description=description)
  
  parser.add_argument('--g', type=str, help='gamma', default = .95)
  parser.add_argument('--bs', type=int, help='batch size of tasks per meta iteration', default = 20)
  parser.add_argument('--h', type=int, help='episode length', default=200)
  parser.add_argument('--nr', type=int, help='number of rollouts per task', default = 2)
  parser.add_argument('--hs', type=int, help='hidden size', default = 100)
  parser.add_argument('--a', type=float, help='alpha', default = 1e-3)
  parser.add_argument('--b', type=float, help='beta', default = 1e-3)
  parser.add_argument('--nmi', type=int, help='numer of meta iterations', default = 20)
  parser.add_argument('--gpu', action='store_true', help='use gpu', default = False)
  parser.add_argument('--gr', type=int, help='range the goal can be in', default = 2)
  parser.add_argument('--asp', type=int, help='action space size', default = 6)
  parser.add_argument('--ssp', type=int, help='state space size', default = 17)
  parser.add_argument('--rew', type=str, help='what type of reward are we using', default = "reward_run")
  args = parser.parse_args()
  return args

def initialize_weights(m):
  if isinstance(m, nn.Linear): #or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
    init.xavier_uniform(m.weight.data)

def select_action(model, state):
  state = torch.from_numpy(state).float().unsqueeze(0)
  means = model(Variable(state))
  std_devs = Variable(torch.FloatTensor(np.array([.5 for i in xrange(6)])))
  action = torch.normal(means, std_devs)
  model.saved_actions.append(action)
  return action.data

def replay_actions(model,states,actions):
  for state,action in zip(actions,states):
    state = torch.from_numpy(state).float().unsqueeze(0)
    output = model(Variable(state))
    model.saved_actions.append(action)

def train_one_step(model,rewards,lr):
  model.train()
  for action, r in zip(model.saved_actions, rewards):
    action.reinforce(r)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  optimizer.zero_grad()
  torch.autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
  optimizer.step()
  del model.rewards[:]
  del model.saved_actions[:]
  return model


def calcDiscountedRewards(rewards,gamma):
  discounted_rewards = []
  future_rewards = 0
  for i in range(len(rewards))[::-1]:
    future_rewards +=rewards[i]
    discounted_rewards.append(future_rewards)
    future_rewards *= gamma
  #discounted_rewards = (np.array(discounted_rewards) - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + np.finfo(np.float32).eps)
  np.array(discounted_rewards)
  return discounted_rewards[::-1]

def policy_gradient_rollouts(model,env,num_rollouts,goal_state,gamma,horizon,reward_type,std_devs):
  states = []
  discounted_rewards = []
  actions = []
  model.eval()
  for rollout in xrange(num_rollouts):
    s = env.reset()
    done = False
    rewards = []
    model.eval()
    for step in xrange(horizon): 
      action = select_action(model,s)
      s,r,done,info = env.step(action.numpy())
      rewards.append(-1.0*abs(goal_state- info[reward_type]))
    discounted_rewards.extend(calcDiscountedRewards(rewards,gamma))
  return states, actions, discounted_rewards

def train_maml(env,meta_model,config):
  #assuming the model vars are initialized and that needed constants are all in config


  for meta_iter in xrange(config.num_meta_iterations):
    test_rollout_states = []
    test_rollout_actions = []
    test_rollout_rewards = []
    for task_number in xrange(config.meta_batch_size):
      meta_model_copy = copy.deepcopy(meta_model)
      goal_state = np.random.rand()*config.goal_range
      #get rollout for task
      std_devs = np.array([.5 for i in xrange(config.action_space_size)])
      states, actions, rewards = policy_gradient_rollouts(meta_model_copy,env,config.num_rollouts,goal_state,config.gamma,config.h,config.reward_type,std_devs)
      rewards = torch.Tensor(rewards)
      if task_number % 10 == 0:
        print "Sum of discounted rewards on training rollouts is " +str(rewards.sum())
      model = train_one_step(meta_model_copy,rewards,config.alpha)

      #get training data for meta learning
      states, actions, rewards = policy_gradient_rollouts(model,env,config.num_rollouts,goal_state,config.gamma,config.h,config.reward_type,std_devs)

      if task_number % 10 == 0:
        print "Sum of discounted rewards on test rollouts is " +str(np.sum(rewards))
      test_rollout_states.extend(states)
      test_rollout_actions.extend(actions)
      test_rollout_rewards.extend(rewards)

    #This does not work!!
    meta_model.train()
    replay_actions(meta_model,test_rollout_states,test_rollout_actions)
    meta_model = train_one_step(meta_model,test_rollout_rewards,config.beta)

  return meta_model


def main():
  args = parseConfig()
  config = Config(args) 
  print(config)
  env = gym.make('HalfCheetah-v1')
  env.reset()
  #model_vars = []
  #model_vars.append(Variable(torch.randn(config.hidden_size,config.state_space_size).type(config.dtype), requires_grad=True))
  #model_vars.append(Variable(torch.randn(config.action_space_size,config.hidden_size).type(config.dtype), requires_grad=True))
  model = PolicyNetwork()
  model.apply(initialize_weights)
  maml = train_maml(env, model,config)



if __name__ == '__main__':
  main()