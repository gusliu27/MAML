# Python Imports
import argparse
import numpy as np
# Torch Imports
import torch
#from torch import LongTensor
#from torchtext.vocab import load_word_vectors
from torch.autograd import Variable
#from torch.nn.utils.rnn import pack_padded_sequence#, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch import cuda, FloatTensor
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler

# Our modules
from models import *
from utils import *
import copy

def initialize_weights(m):
  if isinstance(m, nn.Linear): #or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
    init.xavier_uniform(m.weight.data)

def train_one_step(model_vars, states, actions,rewards,lr):
  #pretty sure we have to do this. :()
  total_grads = []
  for var in model_vars:
    total_grads.append(0)
  for s_idx in xrange(len(states)):
    state = tensor(states[s_idx])
    for var in model_vars:
      state = var.mm(state).clamp(min=0)
    #Now state is the value of our forward prop, so we computed the appropriate activations
    action_activations = state.data.numpy()
    log_probabilities = np.log(action_activations/np.sum(action_activations))
    dlog_probabilities = -1*log_probabilities
    dlog_probabilities[actions[s_idx]] += 1
    reward_times_dlogprobs = rewards[s_idx]
    torch.autograd.backward(model_vars[len(model_vars)-1])
    #collect all the gradients from each example in total grads, then we will return them
    for var_idx in xrange(len(model_vars)):
      total_grads[i] += model_vars[var_idx].grad.data
      model_vars[var_idx].grad.data.zero_()
  for var_idx in xrange(len(model_vars)):
    model_vars[var_idx] -= lr*total_grads[var_idx]
  return model_vars

def calcDiscountedRewards(rewards,gamma):
  discounted_rewards = []
  future_rewards = 0
  for i in range(len(rewards))[::-1]:
    future_rewards +=rewards[i]
    discounted_rewards.append(future_rewards)
    future_rewards *= gamma
  return discounted_rewards[::-1]

def policy_gradient_rollouts(model_vars,env,num_rollouts,reward_fn,gamma,horizon):
  states = []
  discounted_rewards = []
  actions = []
  model.eval()
  for rollout in xrange(num_rollouts):
    s = env.reset()
    done = False
    rewards = []
    for step in xrange(horizon): 
      a = 0
      states.append(s)
      for var in model_vars:
        s = var.mm(s).clamp(min=0)
      #Now state is the value of our forward prop, so we computed the appropriate activations
      action_activations = state.data.numpy()
      sum_prob = np.sum(action_activations)
      runningProb = 0
      rand = np.random.rand()
      #THIS NEEDS TO CHANGE DEPENDING ON HOW WE GET PROBABILITIES FROM NETWORK
      for i in xrange(len(action_activations)):
        runningProb += action_activations[i]/sum_prob
        if rand < runningProb:
          a = i
          break
      actions.append(a)
      rewards.append(-1.0*abs(reward_fn - velocity)) #how do i know my velocity??
      s,r,done,info = env.step(a)
    discountedRewards.append(calcDiscountedRewards(rewards,gamma))
  return states, actions, discountedRewards

def train_maml(env,meta_model_vars,config):
  #assuming the model vars are initialized and that needed constants are all in config


  for meta_iter in xrange(config.num_meta_iterations):
    test_rollout_states = []
    test_rollout_actions = []
    test_rollout_rewards = []
    for task in tasks:
      reward_fn = np.random.rand()*3
      #get rollout for task
      states, actions, rewards = policy_gradient_rollouts(model_vars,env,config.num_rollouts,reward_fn,config.gamma,config.h)
      model = train_one_step(model_vars, states, actions,rewards,config.alpha)
      #get training data for meta learning
      states, actions, rewards = policy_gradient_rollouts(model,env,config.num_rollouts,reward_fn,config.gamma,config.h)
      test_rollout_states.append(states)
      test_rollout_actions.append(actions)
      test_rollout_rewards.append(rewards)

    #do meta learning here
    train_one_step(meta_model_vars, test_rollout_states, test_rollout_actions,test_rollout_rewards,config.beta)

  return meta_model_vars


  

if __name__ == '__main__':
  main()