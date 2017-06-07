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
    self.vanilla_episodes = args.ep


  def __str__(self):
    properties = vars(self)
    properties = ["{} : {}".format(k, str(v)) for k, v in properties.items()]
    properties = '\n'.join(properties)
    properties = "--- Config --- \n" + properties + "\n"
    return properties


def parseConfig(description="Default Model Description"):
  parser = argparse.ArgumentParser(description=description)
  
  parser.add_argument('--g', type=str, help='gamma', default = .99)
  parser.add_argument('--bs', type=int, help='batch size of tasks per meta iteration', default = 40)
  parser.add_argument('--h', type=int, help='episode length', default=200)
  parser.add_argument('--nr', type=int, help='number of rollouts per task', default = 20)
  parser.add_argument('--hs', type=int, help='hidden size', default = 100)
  parser.add_argument('--a', type=float, help='alpha', default = 1e-3)
  parser.add_argument('--b', type=float, help='beta', default = 1e-3)
  parser.add_argument('--nmi', type=int, help='numer of meta iterations', default = 100)
  parser.add_argument('--gpu', action='store_true', help='use gpu', default = False)
  parser.add_argument('--gr', type=int, help='range the goal can be in', default = 2)
  parser.add_argument('--asp', type=int, help='action space size', default = 6)
  parser.add_argument('--ssp', type=int, help='state space size', default = 17)
  parser.add_argument('--rew', type=str, help='what type of reward are we using', default = "reward_run")
  parser.add_argument('--ep', type=str, help='how many episodes to train the vanilla network, if we are training one for comparison to MAML', default = 100)
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

def train_one_step(model,rewards,lr,doUpdate = True):
  model.train()
  for action, r in zip(model.saved_actions, rewards):
    action.reinforce(r)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  optimizer.zero_grad()
  torch.autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
  if doUpdate:
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
  avg_loss_per_metaiter = []
  for meta_iter in xrange(config.num_meta_iterations):
    #test_rollout_states = []
    #test_rollout_actions = []
    #test_rollout_rewards = []
    total_loss = 0
    new_model = copy.deepcopy(meta_model)
    print "meta iteration number " + str(meta_iter)
    goal_state = 0
    for task_number in xrange(config.meta_batch_size):
      meta_model_copy = copy.deepcopy(meta_model)
      goal_state += 1.0*config.goal_range/config.meta_batch_size
      #get rollout for task
      std_devs = np.array([.1 for i in xrange(config.action_space_size)])
      states, actions, rewards = policy_gradient_rollouts(meta_model_copy,env,config.num_rollouts,goal_state,config.gamma,config.h,config.reward_type,std_devs)
      total_loss += np.sum(rewards)
      rewards = torch.Tensor(rewards)
      if task_number % 10 == 0:
        print "goal_state is " + str(goal_state)
        print "Sum of discounted rewards on training rollouts is " +str(rewards.sum())
      train_one_step(meta_model_copy,rewards,config.alpha)

      #get training data for meta learning
      states, actions, rewards = policy_gradient_rollouts(meta_model_copy,env,config.num_rollouts,goal_state,config.gamma,config.h,config.reward_type,std_devs)
      train_one_step(meta_model_copy,rewards,config.alpha,False)
      #meta learning
      for copy_param, new_model_param in zip(meta_model_copy.parameters(),new_model.parameters()):
        new_model_param.data -= config.beta*copy_param.grad.data
      if task_number % 10 == 0:
        print "Sum of discounted rewards on test rollouts is " +str(np.sum(rewards))

      #test_rollout_states.extend(states)
      #test_rollout_actions.extend(actions)
      #test_rollout_rewards.extend(rewards)
    print "average loss from test samples on meta iteration " +str(meta_iter) + " was " + str(total_loss/(config.meta_batch_size*config.num_rollouts))
    avg_loss_per_metaiter.append(total_loss/(config.meta_batch_size*config.num_rollouts))
    meta_model = new_model
  print avg_loss_per_metaiter
  return meta_model


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

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
  
  #use this if you want to save the maml!! then you wont have to compute it over and over
  save_checkpoint({'state_dict': maml.state_dict()},filename="maml.checkpoint.tar")
  loaded_maml = PolicyNetwork()
  checkpoint = torch.load("maml.checkpoint.tar")
  loaded_maml.load_state_dict(checkpoint['state_dict'])


  #initialize a network with policy gradient for comparison
  vanilla_network = PolicyNetwork()

  goal_state = 1.0 #randomly chosen
  std_devs = np.array([.1 for i in xrange(config.action_space_size)])
  for ep in xrange(config.vanilla_episodes):
    states, actions, rewards = policy_gradient_rollouts(vanilla_network,env,config.num_rollouts,goal_state,config.gamma,config.h,config.reward_type,std_devs)
    if ep%10 == 0:
      print "Vanilla network total rewards is " + str(np.sum(rewards))
    rewards = torch.Tensor(rewards)
    train_one_step(vanilla_network,rewards,config.alpha)



  #compare the networks!

  test_goal_states = [.48,1.02,1.44] #randomly chosen goal states
  for test_goal_state in test_goal_states:
    print "comparing networks on goal state of " + str(test_goal_state)
    vanilla_network_copy = copy.deepcopy(vanilla_network)
      #initialize a random network for comparison
    random_model = PolicyNetwork()

    maml_rewards = []
    random_rewards = []
    pretrained_rewards = []
    for i in xrange(10):#how many times do we want to train the thing on new data??
      states, actions, rewards = policy_gradient_rollouts(vanilla_network_copy,env,config.num_rollouts,test_goal_state,config.gamma,config.h,config.reward_type,std_devs)
      print "sum of the rewards for vanilla after " + str(i) +" steps: " + str(np.sum(rewards))
      pretrained_rewards.append(np.sum(rewards))
      train_one_step(vanilla_network_copy,rewards,config.alpha)

      states, actions, rewards = policy_gradient_rollouts(maml,env,config.num_rollouts,test_goal_state,config.gamma,config.h,config.reward_type,std_devs)
      print "sum of the rewards for maml after " + str(i) +" steps: " + str(np.sum(rewards))
      maml_rewards.append(np.sum(rewards))
      train_one_step(maml,rewards,config.alpha)

      states, actions, rewards = policy_gradient_rollouts(random_model,env,config.num_rollouts,test_goal_state,config.gamma,config.h,config.reward_type,std_devs)
      print "sum of the rewards for random model after " + str(i) +" steps: " + str(np.sum(rewards))
      random_rewards.append(np.sum(rewards))
      train_one_step(random_model,rewards,config.alpha)
    print maml_rewards
    print random_rewards
    print pretrained_rewards



if __name__ == '__main__':
  main()