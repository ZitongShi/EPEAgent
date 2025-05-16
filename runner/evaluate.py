
import pandas as pd, json
from fold1.profile_utils import find_answer_letter

def evaluate(log_path: str, gold_csv: str):
    gold = pd.read_csv(gold_csv, names=['Q','(a)','(b)','(c)','(d)','Ans'])['Ans']
    results = json.load(open(log_path, 'r', encoding='utf-8'))
    preds = [find_answer_letter(r.get('response', r.get('sum_response', ''))) for r in results]
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    print('Test Accuracy:', acc)
