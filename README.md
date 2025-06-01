# 🎮 🤖 QMLE: Q-Learning with MLE


[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/atavakol/qmle)


This repository contains the official code release for the paper [*Learning in complex action spaces without policy gradients*](https://openreview.net/forum?id=nOL9M6D4oM) by Arash Tavakoli, Sina Ghiassian, and Nemanja Rakićević.
The paper is published in [*Transactions on Machine Learning Research (TMLR)*](https://jmlr.org/tmlr/).

This implementation was developed by [Arash Tavakoli](https://atavakol.github.io/).


### Citation

Please use the following citation if you make use of our work:

```
@article{tavakoli2025learning,
  author  = {Arash Tavakoli and Sina Ghiassian and Nemanja Rakicevic},
  title   = {Learning in complex action spaces without policy gradients},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  url     = {https://openreview.net/forum?id=nOL9M6D4oM}
}
```


## Get Started

To run experiments locally, give the following a try:
```bash
git clone https://github.com/atavakol/qmle.git && cd qmle
pip install -e .
```

Start training with capture video:
```bash
python -m src.qmle --env-id walker_stand --capture-video
```

### Attributions

This project includes modified code from the following repositories:

- [CleanRL](https://github.com/vwxyzjn/cleanrl) - DQN implementation, licensed under MIT.
- [Stable Baselines](https://github.com/hill-a/stable-baselines) - Prioritized Replay Buffer, licensed under MIT.

Each respective license is included in the `third_party/` directory.
