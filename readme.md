# RL4RS

Reinforcement learning based experiments for recommender systems.

## Description

This repository is based on 'Interactive Recommendation with User-Specific Deep Reinforcement Learning' by Yu Lei and Wenjie Li. It is not intended to be a precise reproduction, but rather an experimental testbed for further approaches.

### Agents

Currently there are two classes of agents implemented:

#### Random Agent

Randomly selects an action from the user's MDP (i.e., a potential item to recommend) and displays it to the user, obtaining a reward. The selection is fully random with no further considerations.

#### DQN Agent

Selects actions from the user's MDP based on a deep q-value neural network. The agent is trained on embeddings of users and items which are obtained using matrix factorization from the train portion of the dataset.

##### Procedure Outline:
The procedure used to train the DQN-agent is outlined below:
1) Creates an interaction matrix R_ui for each user u and item i in the dataset
2) Uses Singular Value Decomposition (SVD) to encode the interaction matrix R_ui into a smaller embedding space (corresponding to the emb_dim parameter). The column vectors of V constitue the item representations of the individual items present in the dataset. 
3) Uses the pre-trained item embedding vectors (i.e., the columns of V) to update the current user state for each step in the episode
4) The agent receives the user embedding as input and generates q-values for the different items
5) The action corresponding to the maximum q-value is taken by the agent, the latent user vector s_t is recomputed with the chosen action
6) The process repeats from step 4. until the specified number of training iterations is reached

## Environment

The environment for each user consists of the items that the user has interacted with and rated in the past. 
Items without ratings are (currently) not included in the potential items to recommend for a user. 
The interaction sequence (s_t) contains the required information for the markov property, as it is a condensation of 
the items interacted with in a session.

## Example Runs

#### Using default parameters and --agent_type='rand':
In fold 1: Mean 3.6309625668449197\
In fold 2: Mean 3.6276699770817418\
In fold 3: Mean 3.6257066462948817\
In fold 4: Mean 3.6265393430099313\
In fold 5: Mean 3.6301757066462947\
Avg. Mean: 3.6282108479755535, Std.: 0.002039123466755497

#### Using default parameters and --agent_type ='dqn' and --num_iterations=1000:
Avg.Reward: 4.158623112984618: 100%|██████████| 1000/1000 [07:27<00:00,  2.24it/s]\
In fold 1: Mean 4.110695499707774\
Avg.Reward: 4.167770831350056: 100%|██████████| 1000/1000 [07:35<00:00,  2.19it/s]\
In fold 2: Mean 4.1885973115137345\
Avg.Reward: 4.1605942488387075: 100%|██████████| 1000/1000 [07:31<00:00,  2.22it/s]\
In fold 3: Mean 4.164424313267095\
Avg.Reward: 4.0647875065996955: 100%|██████████| 1000/1000 [07:21<00:00,  2.26it/s]\
In fold 4: Mean 4.003389830508475\
Avg.Reward: 4.063753918000273: 100%|██████████| 1000/1000 [07:23<00:00,  2.26it/s]\
In fold 5: Mean 4.024336645236704\
Avg. Mean: 4.098288720046757, Std.: 0.07369965423151358

#### Using default parameters and --agent_type ='dqn' and --num_iterations=5000:
Avg.Reward: 4.1422340829630935: 100%|██████████| 5000/5000 [36:56<00:00,  2.26it/s]\
In fold 1: Mean 4.153208649912332\
Avg.Reward: 4.055631633843068: 100%|██████████| 5000/5000 [36:28<00:00,  2.28it/s]\
In fold 2: Mean 4.183664523670368\
Avg.Reward: 4.112256386006553: 100%|██████████| 5000/5000 [37:12<00:00,  2.24it/s]\
In fold 3: Mean 4.193278784336645\
Avg.Reward: 4.060550415977765: 100%|██████████| 5000/5000 [37:09<00:00,  2.24it/s]\
In fold 4: Mean 4.189672706019872\
Avg.Reward: 4.079064650456784: 100%|██████████| 5000/5000 [39:15<00:00,  2.12it/s]\
In fold 5: Mean 4.134488603156049\
Avg. Mean: 4.170862653419053, Std.: 0.023043108550898885

## References

Yu Lei and Wenjie Li. 2019. Interactive Recommendation with User-Specific Deep Reinforcement Learning. ACM Trans. Knowl. Discov. Data 13, 6, Article 61 (December 2019), 15 pages. DOI:https://doi.org/10.1145/3359554