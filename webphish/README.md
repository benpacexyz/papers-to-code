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
* The code in this repo should be run as step_00 -> step01.

## Model Scoring Metrics Overview

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
