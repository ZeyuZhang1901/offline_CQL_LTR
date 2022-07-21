## Result

### 7.19

see  results at  [cql results](https://wandb.ai/zeyuzhang/SimpleSAC--sac/runs/7128548fd2474483beb69e9ea767615d?workspace=user-zeyuzhang) 

### 7.21

The performance might collapse if we roll out while training policy at the same time. Thus we choose to generate 1e6 samples before training start this time, and these are the results: [cql_results_no_rollout](https://wandb.ai/zeyuzhang/SimpleSAC--sac/runs/736a02bb20274e3a8abadca044108214) 