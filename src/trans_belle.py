import json

def convert_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        new_data_list = []
        for index, line in enumerate(infile):
            # 加载原始数据行
            original_data = json.loads(line)

            # 新格式的数据结构初始化
            new_data = {
                "instruction": "",
                "input": "",
                "output": "",
                "system": "",
                "history": []
            }
            splited_data = original_data["instruction"].split("Human:")[1:]
            for history_index in range(len(splited_data)-1):
                history_elem = splited_data[history_index].split("Assistant:")
                try:
                    new_data["history"].append([history_elem[0][:-1],history_elem[1][:-1]])
                except:
                    print(index)
                    continue
            new_data["instruction"] = splited_data[-1][:-11]
            new_data["output"] = original_data["output"]
            if len(new_data["history"]) > 1:
                new_data_list.append(new_data)
                
        json.dump(new_data_list, outfile, ensure_ascii=False)

convert_format(r'J:\AI\datasets\multiturn_chat_1024-4096.json', 'chat_p2.json')
