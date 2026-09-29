# CIFAR-100 运行时间预估摘要

- GPU：NVIDIA GeForce RTX 3090
- 机器数：4
- 保留比例：[60, 70, 80, 90]
- 排名口径：生成四个目标保留比例 mask 的总墙钟时间

| 排名 | 方法 | 总时间 | 主要耗时阶段 | 对应比例 | 阶段占比 | 状态 |
|---:|---|---:|---|---:|---:|---|
| 1 | Random | 0.00秒 | sample_and_build_four_masks | 共享 | 100.0% | ok |
| 2 | Herding | 4分 15.1秒 | classwise_greedy_selection | 90% | 25.5% | ok |
| 3 | EL2N | 5分 00.1秒 | proxy_training_to_score_epoch | 共享 | 98.9% | ok |
| 4 | GraNd | 7分 39.0秒 | proxy_training_to_score_epoch | 共享 | 58.3% | ok |
| 5 | MDS | 19分 01.3秒 | resnet50_proxy_training | 共享 | 99.4% | ok |
| 6 | YangCLIP | 20分 57.3秒 | clip_adapter_training | 共享 | 72.2% | ok |
| 7 | Forgetting | 45分 59.1秒 | proxy_training_with_forgetting_tracking | 共享 | 100.0% | ok |
| 8 | RLSelector | 2小时 31分 | rl_guided_training | 80% | 23.7% | ok |
| 9 | MoSo | 4小时 00分 | ten_checkpoint_exact_gradient_scoring | 60% | 22.3% | ok |
