import pandas as pd
import re
# 从原始文本中解析出需要的字段
def parse_text(text):
    rows = []
    row={}
    for line in text.split("\n"):
        if line.startswith("CRITICAL:root:"):
            parts = line.split(":")
            key = parts[2]
            value =''.join( parts[3:])
            if key in  ["time"]:
                row[key]=value
            if key=="args":
                pattern = r"([a-zA-Z_]+)=('[^']+'|[\d\.]+)"
                matches = re.findall(pattern, value)
                for match in matches:
                    row[match[0]] =  eval(match[1])
            if key=="best_val_result":
                pattern =r"'([\da-zA-Z_]+)' [a-zA-Z_]+?[\(]?([\d.]+)"
                matches = re.findall(pattern, value)
                pattern =r"'([\da-zA-Z_]+)' ([\d.]+)"
                matches+= re.findall(pattern, value)
                for match in matches:
                    row["best_val_"+str(match[0])] =  round(eval(match[1])*100, 2)
            if key=="best_test_result":
                pattern =r"'([\da-zA-Z_]+)' [a-zA-Z_]+?[\(]?([\d.]+)"
                matches = re.findall(pattern, value)
                pattern =r"'([\da-zA-Z_]+)' ([\d.]+)"
                matches+= re.findall(pattern, value)
                for match in matches:
                    row["best_test_"+match[0]] =  round(eval(match[1])*100, 2)
            if key=="best_test_result":
                rows.append(row)
                row = {}
                
    return pd.DataFrame(rows)

# 读取原始文本并解析成 CSV 格式
def main():
    with open("log.txt", "r", encoding="utf-8") as f:
        text = f.read()
        data = parse_text(text).dropna(axis=0,how='any')
    data.to_csv("output_log.csv", index=False)
    clean=data.groupby(['dataset','lp_model','exp_type','attack_method','attack_goal','attack_rate'])[['best_val_auc', 'best_val_acc', 'best_val_recall', 'best_val_precision',
       'best_val_f1', 'best_test_auc', 'best_test_acc', 'best_test_recall',
       'best_test_precision', 'best_test_f1']].mean()
    clean=clean[['best_val_auc', 'best_test_auc', 'best_val_auc', 'best_val_acc', 'best_val_recall', 'best_val_precision',
       'best_val_f1', 'best_test_auc', 'best_test_acc', 'best_test_recall',
       'best_test_precision', 'best_test_f1']].apply(lambda x:round(x,2))
    clean=clean.apply(lambda x:round(x,2))
    clean.to_csv('output_result.csv')

if __name__ == "__main__":
    main()