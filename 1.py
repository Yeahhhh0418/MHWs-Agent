import json

# 需要检查的字段列表
FIELDS_TO_CHECK = [
    "EnemyExtraName",
    "EnemyBossExp",
    "EnemyFrenzyName",
    "EnemyLegendaryName",
    "EnemyLegendaryKingName",
    "EnemyTips"
]

def clean_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # 删除值为空字符串的指定字段
            for field in FIELDS_TO_CHECK:
                if field in data and data[field] == "":
                    del data[field]
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_path = "C:\\code\\MHWI_agent\\data\\Monster_info_Table.jsonl"   # 输入文件路径
    output_path = "C:\\code\\MHWI_agent\\Monster_info_Table.jsonl" # 输出文件路径
    clean_jsonl(input_path, output_path)
    print("处理完成！")