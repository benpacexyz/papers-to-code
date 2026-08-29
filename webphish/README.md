# WebPhish

This repo is a demo project that showcases an implementation of the Convolutional Neural 
Network model outlined in the paper 
[Look before you leap: Detecting phishing web pages by exploiting raw URL and HTML characteristics](https://www.sciencedirect.com/science/article/pii/S0957417423016858)

**Paper Authors:** Chidimma Opara, Yingke Chen, Bo Wei\
**Repo Author:** Benjamin Pace

## Project Notes

* The data files prefixed `collected_xx` were collected from the following sources. These data were collected with an automated web scraper that ran on Kali Linux in VirtualBox for isolation.
  * [PhishTank](https://phishtank.org/) - Repository and catalog of known phishing websites collected and maintained by Cisco.
  * [Aung et al. Paper](https://github.com/ESDAUNG/PhishDataset) - A collection of approximately 20,000 websites that are approximately distributed as 50% malicious (phishing) and 50% benign websites. These data were collected and published as part of a research paper for the ACM International Conference on Web Intelligence.
* The data files prefixed `webphish_xx`were published by the "Look before you leap" authors.
* The code in this repo should be run as step_00 -> step_01 -> step_02.

## Optimization (Auto Research)

The first iteration of the model step_01 closely follows the implementation in the research paper. However, the model was optimized past step_01 through an automated research process (i.e. see karpathy's auto-research) in which an agent was instructed to conduct a structured hyperparameter and architecture search using Optuna. As shown in the prompt below the agent was also instructed to evaluate candidate configurations against a multi-seed baseline to distinguish genuine improvements from run-to-run variance.

The best-performing configuration was then human validated through repeated runs to confirm the stability of the reported results.

Codex GPT 5.6 Sol (max) was given 3 independent attempts to improve the results and the highest scoring model based on a final hold out set is the final model shown in step_02. This README.md and the final holdout set were omitted from the files available to the agent.

```
I'm trying to improve the results of the model trained in src/step_01_training.py. Testing results currently seem a bit lower than what is acceptable for the use cases of this model.

Rerun the current config 5 times with different seeds to check the baseline variance. If the expected results fall within that range, the drop may just be noise, so flag that before tuning anything.

Then make adjustments to the architecture and hyperparameters to improve the results. You can use Optuna to try out different parameters and tune the model. Keep it straightforward, explore simple and complex adjustments, but remeber we must not overfit. Its very important that we ensure the model generalizes well outside of this tuning process.

You may not edit the existing code, but you can write and run as many scratch scripts as you want in the agent/ folder, or run scripts inline. Use the python .venv in the current root. For context, the paper the model is based on (not a direct copy) is in paper/2024_Expert Systems with Applications_Look Before You Leap.pdf if you want to review it. You should also only use the existing html_test.parquet and html_train.parquet files, you may not regenerate those. The file src/step_00_train_test_split.py is only included for context and html_valid.parquet is omitted from the current dir.

As a final step, run the best config a few times without fixed seeds and report the mean and spread, so we can verify the results are stable.

Once you complete this and identify material improvements, please write out your suggested updates in an UPDATE.md file. Keep your report straightforward and simple, and explain the intuition behind your suggested updates.
```

## Model Scoring Metrics Overview

step_01
```
Training WebPhish Results:
  Accuracy: 0.9972
  Precision: 0.9923
  Recall: 0.9989
  F1-Score: 0.9956
  False Positive Rate (FPR): 0.0036
  False Negative Rate (FNR): 0.0011

Testing WebPhish Results:
  Accuracy: 0.9606
  Precision: 0.9665
  Recall: 0.9542
  F1-Score: 0.9603
  False Positive Rate (FPR): 0.0329
  False Negative Rate (FNR): 0.0458
```

step_02
```
Training WebPhish Results:
  Accuracy: 0.9972
  Precision: 0.9923
  Recall: 0.9989
  F1-Score: 0.9956
  False Positive Rate (FPR): 0.0036
  False Negative Rate (FNR): 0.0011

Testing WebPhish Results:
  Accuracy: 0.9606
  Precision: 0.9665
  Recall: 0.9542
  F1-Score: 0.9603
  False Positive Rate (FPR): 0.0329
  False Negative Rate (FNR): 0.0458
```