import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import torch

import absl.app
import absl.flags

from .conservative_sac import ConservativeSAC
# from .sac import SAC
from .replay_buffer import ReplayBuffer, batch_to_torch
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger

from .PretrainRanker.run_MDP_cascade import run
from .PretrainRanker.dataset.Preprocess import PreprocessDataset
from .PretrainRanker.Ranker.MDPRankerV2 import MDPRankerV2
from .PretrainRanker.Click_Model.CM_model import CM
from .PretrainRanker.utils import evl_tool


FLAGS_DEF = define_flags_with_default(
    max_traj_length=10,  # max of the length is 10
    replay_buffer_size=1000000,
    seed=42,
    device='cuda',
    save_model=True,

    # reward_scale=1.0,
    # reward_bias=0.0,
    # clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=2000,
    bc_epochs=0,
    n_total_sample=1000000,    # num of total samples  
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    # online pretrain
    batch_size=256,
    doc_feature_size = 46,  # MQ2007 dataset
    online_lr = 0.001,
    online_eta = 1,
    online_num_iteration=10000,
    click_model = 'informational',  # "perfect", "informational", "navigational"
    reward_method = 'both', # "positive", "negative", "both"

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def offline_train(argv):
    '''prepare wandb logger'''
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    '''pretrain ranker and prepare offline click data'''
    print("***************************************************************")
    print("Online training start!")
    offlinedatapath = "./offlinedata.txt"
    doc_len, dataset, clicks_dataset = Pretrain(FLAGS)
    # OfflineDataDisplay(clicks_dataset, offlinedatapath)
    print("Offline data prepared! Details in "+offlinedatapath+"\n")
    print("Online training finish!\n")

    '''offline train start'''
    print("***************************************************************")
    set_random_seed(FLAGS.seed)

    train_sampler = StepSampler(doc_len, dataset, clicks_dataset, FLAGS.max_traj_length)
    eval_sampler = TrajSampler(doc_len, dataset, clicks_dataset, FLAGS.max_traj_length)

    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)

    policy = TanhGaussianPolicy(
        eval_sampler.observation_dim,
        eval_sampler.action_dim,
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunction(
        eval_sampler.observation_dim,
        eval_sampler.action_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        eval_sampler.observation_dim,
        eval_sampler.action_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod((eval_sampler.action_dim,)).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    '''prepare click dataset'''
    print("Prepare click dataset in replay buffer:\n")
    print("Initial parameters in policy: ")
    print(policy.parameters())
    print("\n")
    train_sampler.sample(
        sampler_policy, FLAGS.n_total_sample,
        deterministic=False, replay_buffer=replay_buffer
    )
    print("Click dataset prepared!\n")
    print("Offline training Start!\n")

    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {}
        # with Timer() as rollout_timer:
        #     train_sampler.sample(
        #         sampler_policy, FLAGS.n_env_steps_per_epoch,
        #         deterministic=False, replay_buffer=replay_buffer
        #     )
        #     metrics['env_steps'] = replay_buffer.total_steps
        #     metrics['epoch'] = epoch
        metrics['env_steps'] = replay_buffer.total_steps
        metrics['epoch'] = epoch

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = batch_to_torch(replay_buffer.sample(FLAGS.batch_size), FLAGS.device)
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac')
                    )
                else:
                    sac.train(batch, bc=epoch < FLAGS.bc_epochs)

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards'])/t['i_reward'] for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])

                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')
    print("\n Offline training finish!")


def Pretrain(FLAGS):
    '''pretrain MDPRanker using online MDP method'''
    dataset_fold = "E:\VScode\VScode_python\Offline_CQL_LTR\SimpleSAC\PretrainRanker\dataset\MQ2007"
    training_path = "{}/Fold1/train.txt".format(dataset_fold)
    test_path = "{}/Fold1/test.txt".format(dataset_fold)
    train_set = PreprocessDataset(training_path,
                                 FLAGS.doc_feature_size,
                                 query_level_norm=False)
    test_set = PreprocessDataset(test_path,
                                FLAGS.doc_feature_size,
                                query_level_norm=False)
    ranker = MDPRankerV2(256,
                        FLAGS.doc_feature_size,
                        FLAGS.online_lr,
                        loss_type='pairwise')
    model_type = FLAGS.click_model
    if model_type == "perfect":
        pc = [0.0, 0.5, 1.0]
        ps = [0.0, 0.0, 0.0]  
    elif model_type == "navigational":
        pc = [0.05, 0.5, 0.95]
        ps = [0.2, 0.5, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.7, 0.9]
        ps = [0.1, 0.3, 0.5]
    cm = CM(pc, ps)

    run(train_set=train_set, 
        test_set=test_set, 
        ranker=ranker,
        eta=FLAGS.online_eta,
        reward_method=FLAGS.reward_method,
        num_interation=FLAGS.online_num_iteration,
        click_model=cm)

    '''prepare offline training data'''
    query_set = train_set.get_all_querys()
    clicks_dataset={}
    doc_len = 0
    for i in range(query_set.shape[0]):
        qid = query_set[i]
        len_q = len(train_set.get_candidate_docids_by_query(qid))
        if len_q>=doc_len:
            doc_len = len_q

        result_list = ranker.get_query_result_list(train_set, qid)
        clicked_doces, click_labels, obs_probs, _ = cm.simulate_cascade(
            query = qid,
            result_list = result_list,
            dataset = train_set,
            k=10
        )
        clicks_dataset[qid] = {"clicked_doces":clicked_doces,
                                "click_labels":np.array(click_labels),
                                "propensities":obs_probs}

    return doc_len, train_set, clicks_dataset

def OfflineDataDisplay(dataset, filepath):
    keys = list(dataset.keys())
    with open(filepath,'w') as f:
        for i in range(len(keys)):
            f.write("qid:"+str(keys[i])+"\n")
            f.write("\tclicked_documents: "+str(dataset[keys[i]]["clicked_doces"])+"\n")
            f.write("\tclicked_labels: "+str(dataset[keys[i]]["click_labels"])+"\n")
            f.write("\tpropensities: "+str(dataset[keys[i]]["propensities"])+"\n")
        

if __name__ == '__main__':
    absl.app.run(offline_train)

