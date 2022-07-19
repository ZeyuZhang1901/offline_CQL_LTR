import numpy as np

class StepSampler(object):

    def __init__(self, doc_len, dataset, clicks_dataset, max_traj_length=10):
        self.max_traj_length = max_traj_length
        self.dataset = dataset
        self.queryset = dataset.get_all_querys()
        self.clicks_dataset = clicks_dataset
        self._current_observation = None

        # initial observation dim and action dim
        self.action_dim = doc_len
        self.observation_dim = 46
        self.start = True
        self.position = 0
        self.mask = np.zeros(self.action_dim)
        self.qid = None
        self.total_step = None

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        prob_dists = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps): 

            # observation (state)
            if self.start:  # start a new query (MDP)
                self.start = False
                self.position = 0
                self.qid = self.queryset[np.random.randint(len(self.queryset))]
                docid_list = self.dataset.get_candidate_docids_by_query(self.qid)
                ndoc = len(docid_list)
                self.total_step = np.min([ndoc,self.max_traj_length])
                self.mask = np.zeros(self.action_dim)  # mask initialization
                self.mask[:ndoc] = 1
                
                docid = np.random.choice(self.action_dim,1,p=self.mask/np.sum(self.mask))[0]  # the first doc is randomly selected
                observation = np.array(self.dataset.get_features_by_query_and_docid(self.qid, docid))
                if docid in self.clicks_dataset[self.qid]["clicked_doces"]:  # first reward
                    reward = 1/np.log2(self.position+2)
                else:
                    reward = 0
                rewards.append(reward)
                self.mask[docid]=0  # mask chosen doc
            else:
                observation = self._current_observation
            observations.append(observation)
            
            # policy and next action (pdf of remain docs)
            prob_dist = policy(
                np.expand_dims(observation, 0), deterministic=deterministic
            )[0, :]
            prob_dist = prob_dist * self.mask  #mask the chosen docs
            prob_sum = np.sum(prob_dist)
            if prob_sum==0:  # if no docs, suppose uniform distribution over remaining docs
                prob_dist[:ndoc] = 1
                prob_dist * self.mask
                prob_dist = prob_dist/np.sum(prob_dist)
            else:
                prob_dist = prob_dist/prob_sum
            prob_dists.append(prob_dist)

            # next observation (next state)
            docid = np.random.choice(self.action_dim,1,p=prob_dist)[0]
            next_feature = np.array(self.dataset.get_features_by_query_and_docid(self.qid, docid))
            next_observation = (next_feature + observations[-1]*len(observations))/\
                (len(observations)+1) if self.position<(self.max_traj_length-1)\
                                      else next_feature
            next_observations.append(next_observation)
            self.mask[docid]=0

            # reward (click signal, binary)
            if self.position !=0:
                if docid in self.clicks_dataset[self.qid]["clicked_doces"]:
                    reward = 1/np.log2(self.position+2)
                else:
                    reward = 0
                rewards.append(reward)

            # done signal
            done = True if self.position==(self.total_step-1) else False
            if done:
                self.start = True
            dones.append(done)

            # replay buffer
            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, prob_dist, reward, next_observation, done
                )

            self._current_observation = next_observation
            self.position += 1

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(prob_dists, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    
class TrajSampler(object):

    def __init__(self, doc_len, dataset, clicks_dataset, max_traj_length=10):
        self.max_traj_length = max_traj_length
        self.dataset = dataset
        self.queryset = dataset.get_all_querys()
        self.clicks_dataset = clicks_dataset
        self._current_observation = None

        # initial observation dim and action dim
        self.action_dim = doc_len
        self.observation_dim = 46
        self.mask = np.zeros(self.action_dim)
        self.qid=None
        self.total_step = None

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            prob_dists = []
            rewards = []
            next_observations = []
            dones = []
            i_reward = 0

            for i in range(self.max_traj_length): 
                # observation (state)
                if i == 0:  # start a new query (MDP)
                    self.qid = self.queryset[np.random.randint(len(self.queryset))]
                    docid_list = self.dataset.get_candidate_docids_by_query(self.qid)
                    ndoc = len(docid_list)
                    i_reward = np.sum(1/np.log2(np.arange(2,len(self.clicks_dataset[self.qid]["clicked_doces"])+2)))
                    self.total_step = np.min([ndoc,self.max_traj_length])
                    self.mask = np.zeros(self.action_dim)  # mask initialization
                    self.mask[:ndoc] = 1

                    docid = np.random.choice(self.action_dim,1,p=self.mask/np.sum(self.mask))[0]  # the first doc is randomly selected
                    observation = np.array(self.dataset.get_features_by_query_and_docid(self.qid, docid))
                    if docid in self.clicks_dataset[self.qid]["clicked_doces"]:  # first reward
                        reward = 1
                    else:
                        reward = 0
                    rewards.append(reward)
                    self.mask[docid]=0
                else:
                    observation = self._current_observation
                observations.append(observation)
                
                # policy and next action (pdf of remain docs)
                prob_dist = policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
                prob_dist = prob_dist * self.mask  #mask the chosen docs
                prob_sum = np.sum(prob_dist)
                if prob_sum==0:  # if no docs, suppose uniform distribution over remaining docs
                    prob_dist[:ndoc] = 1
                    prob_dist * self.mask
                    prob_dist = prob_dist/np.sum(prob_dist)
                else:
                    prob_dist = prob_dist/prob_sum
                prob_dists.append(prob_dist)

                # next observation (next state)
                docid = np.random.choice(self.action_dim,1,p=prob_dist)[0]
                next_feature = np.array(self.dataset.get_features_by_query_and_docid(self.qid, docid))
                next_observation = (next_feature + observations[-1]*len(observations))/\
                    (len(observations)+1) if i<(self.max_traj_length-1)\
                                        else next_feature
                next_observations.append(next_observation)
                self.mask[docid]=0

                # reward (click signal, binary)
                if i!=0:
                    if docid in self.clicks_dataset[self.qid]["clicked_doces"]:
                        reward = 1/np.log2(i+2)
                    else:
                        reward = 0
                    rewards.append(reward)

                # done signal
                done = True if i==(self.total_step-1) else False
                dones.append(done)
                
                # replay buffer
                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, prob_dist, reward, next_observation, done
                    )

                self._current_observation = next_observation

                if done:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(prob_dists, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                i_reward=i_reward
            ))

        return trajs
