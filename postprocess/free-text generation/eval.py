import string
import pandas as pd

from builtins import dict
from nltk import sent_tokenize, word_tokenize

from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from spice.spice import Spice
import glob, os

class MIMICEvalCap:
    def __init__(self, pred_path = "/home/guests/mlmi_kamilia/RATCHET/eval_nlp/ratchet_txt/" , gt_path = "/home/guests/mlmi_kamilia/RATCHET/eval_nlp/gt_txt/"):
        pre_ls = []
        for File in os.listdir(pred_path):
            if File.endswith(".txt"):
                pre_ls.append(File)

        gt_ls = []
        for File in os.listdir(gt_path):
            if File.endswith(".txt"):
                gt_ls.append(File)

        self.pred_path = pred_path
        self.gt_path = gt_path
        self.pred_df = pre_ls#pd.read_csv(pred_df_csv, header=None).values
        self.true_df = gt_ls#pd.read_csv(true_df_csv, header=None).values

        self.eval = dict()
        self.imgToEval = dict()

    def preprocess_pred(self, s):
        with open(self.pred_path+s,'r') as f:
            pred = f.read()


        pred = pred.replace('\n', '')
        pred = pred.replace('<s>', '')
        pred = pred.replace('</s>', '')
        # s = s.translate(str.maketrans('', '', '0123456789'))
        # s = s.translate(str.maketrans('', '', string.punctuation))
        return pred

    def preprocess_gt(self, s):
        with open(self.gt_path+s,'r') as f:
            gt = f.read()

        gt = gt.replace('\n', '')
        gt = gt.replace('<s>', '')
        gt = gt.replace('</s>', '')
        # s = s.translate(str.maketrans('', '', '0123456789'))
        # s = s.translate(str.maketrans('', '', string.punctuation))
        return gt

    def evaluate(self):

        gts = dict()
        res = dict()

        # Sanity Checks
        #assert len(self.pred_df) == len(self.true_df)

        # =================================================
        # Pre-process sentences
        # =================================================
        print('tokenization...')
        for i in range(len(self.pred_df)):
            pred_text = ' '.join(word_tokenize(self.preprocess_pred(self.pred_df[i])))
            true_text = ' '.join(word_tokenize(self.preprocess_gt(self.pred_df[i])))

            res[i] = [pred_text]
            gts[i] = [true_text]

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = dict()
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = dict()
        self.imgToEval = dict()
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = dict()
        res = dict()
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = dict()
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


#if __name__ == "__main__":
print("prior:")
eval_mimic = MIMICEvalCap(pred_path = "/home/guests/mlmi_kamilia/RATCHET/eval_nlp/reports_prior/comparison/" , gt_path = "/home/guests/mlmi_kamilia/RATCHET/eval_nlp/gt_txt/comparison/")
eval_mimic.evaluate()
print("ratchet:")
eval_mimic = MIMICEvalCap(pred_path = "/home/guests/mlmi_kamilia/RATCHET/eval_nlp/ratchet_txt/comparsion/" , gt_path = "/home/guests/mlmi_kamilia/RATCHET/eval_nlp/gt_txt/comparison/")
eval_mimic.evaluate()






