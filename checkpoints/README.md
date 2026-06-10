# Model Checkpoints

Pre-trained DQN weights for `MineRLObtainDiamondShovel-v0` (log collection task).

| Directory | Training run | Best file to load |
|-----------|--------------|-------------------|
| `dqn_50k/` | 50,000 timesteps | `dqn_final.pt` |
| `dqn_250k/` | 250,000 timesteps (9k steps saved) | `dqn_step_9000.pt` |

Load a checkpoint in the notebook:

```python
checkpoint = torch.load("checkpoints/dqn_50k/dqn_final.pt", weights_only=False)
```
