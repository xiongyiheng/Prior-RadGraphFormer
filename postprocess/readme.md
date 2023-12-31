# Postprocess for downstream tasks

## Task1: free-text generation and evaluation
Step1: Generate free-text reports from the graph based on manually defined rules using [script](free-text%20generation/generate%20text%20from%20graph.py).

Step2: Evaluate the reports quality regarding NLP metrics with [script](free-text%20generation/eval.py).
This script needs to work with the RATCHET library, you may need put this script in its nlp_metrics folder, so the dir looks like 
<code>RATCHET/nlp_metrics/eval.py</code>

Extra: We provide a demo script to show that NLP metric **sucks** in the clinical application [here](free-text%20generation/comparison.py) ;)

## Task2: classification generation and evaluation
Step1: Given a report, we need to parse it firstly into clinical entities as first step. 
For this, we use [CheXpert](https://github.com/stanfordmlgroup/chexpert-labeler) to extract the entities from the
reports.

Step2: We selected entities of interest out, and evaluate the classification performance by comparing with
the ground truth with [script](classification/eval_ratchet.py).

For the classification results generated by Prior-RadGraphFormer, the performance is evaluated with [inference.py](../inference.py) and [radgraph_eval.py](../util/radgraph_eval.py).
