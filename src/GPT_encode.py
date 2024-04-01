import torch, json, os
from torch.nn.functional import softmax
from rich import print
from rich.progress import Progress
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
#from accelerate import dispatch_model, infer_auto_device_map

model_path = "/root/autodl-tmp/chinese-alpaca-2-13b"
#model_path = r"J:\AI\models\gemma-7b-it"
#lora_path = r"J:\AI\models\gemma-7b-it-lora\chinese-prelora"
add_persent = 0.3
batch_size = 4

class Prompt:
    def __init__(self,tokenizer) -> None:
        self.chat = []
        self.tokenizer = tokenizer
    def add_user_message(self,content):
        self.chat.append({"role": "user", "content": content})
    def add_model_reply(self,content):
        self.chat.append({"role": "assistant", "content": content})
    def build(self):
        return tokenizer.apply_chat_template(self.chat, tokenize=False)

def get_base(prompt,y):
    y_ids = tokenizer.encode(y)
    probsum = 0
    progress_task = progress.add_task("[cyan]具体对话处理进度(准线)", total=len(y_ids))

    prompt_encoded_len = len(tokenizer.encode(prompt))
    encoded_input = tokenizer.encode(prompt + y, return_tensors="pt").to("cuda")

    for y_index, y_id in enumerate(y_ids):
        out = softmax(model.generate(input_ids=encoded_input[0,:prompt_encoded_len+y_index].unsqueeze(0),
                            do_sample=False,
                            output_scores=True,
                            return_dict_in_generate = True,
                            max_new_tokens=1
                            ).scores[0].squeeze(0),dim=0)
        probsum += out[y_id]
        progress.update(progress_task, advance=1)
    progress.remove_task(progress_task)
    print(f"概率和:{probsum}")
    return probsum

def get_diff(base:float,prompt:str,y:str):
    y_ids = tokenizer.encode(y,add_special_tokens=False)
    probsum = 0
    progress_task = progress.add_task("[cyan]具体对话处理进度", total=len(y_ids))

    prompt_encoded_len = len(tokenizer.encode(prompt))
    encoded_input = tokenizer.encode(prompt + y, return_tensors="pt").to("cuda")

    for y_index, y_id in enumerate(y_ids):
        with torch.no_grad():
            out = softmax(model.generate(input_ids=encoded_input[0,:prompt_encoded_len+y_index].unsqueeze(0),
                                do_sample=False,
                                output_scores=True,
                                return_dict_in_generate = True,
                                max_new_tokens=1
                                ).scores[0].squeeze(0),dim=0)
        probsum += out[y_id]
        progress.update(progress_task, advance=1)
    progress.remove_task(progress_task)
    print(f"概率和:{probsum}")
    probdiff = base-probsum
    print(f"概率和差分(越大越有用):{probdiff}")
    return probdiff, probsum

print("加载LLM")
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16,
                                             attn_implementation="flash_attention_2",
                                             )
model = torch.compile(model)
#device_map = infer_auto_device_map(
#    model,
#    max_memory={1: "20GiB","cpu": "10GiB"},
#    no_split_module_classes=["LlamaDecoderLayer"],
#    dtype='float16')
# 输出层设备必须与输入层一致
#print(device_map)
#device_map['model.lm_head'] = device_map['model.embed_tokens']
#model = dispatch_model(model, device_map=device_map)
print("加载Lora")
#model = PeftModel.from_pretrained(model,lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=torch.float16)

model.eval()
model.to("cuda")

with open(r"chat_0.8M.json",'r') as file:
    data = json.loads(file.read())
#获取打分进度
data_old=[]
if os.path.exists("chat_0.8M_encoded.json"):
    with open("chat_0.8M_encoded.json",'r') as file:
        data_old=json.loads(file.read())

with Progress() as progress:
    main_task = progress.add_task("[green]数据集处理进度", total=len(data))

    for block_index, block in enumerate(data):
        if block_index < len(data_old):
            data[block_index] = data_old[block_index]
            progress.update(main_task, advance=1)
            continue

        prompt = Prompt(tokenizer)
        probdiff = []
        probsum = []
        print("计算基准(无修改准线)中")
        for history_elem in block["history"]:
            prompt.add_user_message(history_elem[0])
            prompt.add_model_reply(history_elem[1])
        base_probsum = get_base(prompt.build(),block["output"])
        task_score = progress.add_task("[blue]计算历史得分中...", total=len(block["history"]))
        #枚举移除历史
        for rhistory_index, history_remove in enumerate(block["history"]):
            prompt = Prompt(tokenizer)
            print(f"移除#{block_index}#{rhistory_index}:{history_remove[1][:10]}...")
            temp_history = block["history"].copy()
            temp_history.pop(rhistory_index)
            for history_elem in temp_history:
                prompt.add_user_message(history_elem[0])
                prompt.add_model_reply(history_elem[1])
            prompt.add_user_message(block["instruction"])
            _diff, _sum = get_diff(base_probsum,prompt.build(),block["output"])
            probsum.append(_sum)
            probdiff.append(_diff)
            progress.update(task_score, advance=1)
        progress.remove_task(task_score)
        progress.update(main_task, advance=1)
        print(f"Q:{block['instruction']} A:{block['output']}")
        print(f"影响好历史对:{block['history'][probdiff.index(max(probdiff))]}")
        print(f"影响差历史对:{block['history'][probdiff.index(min(probdiff))]}")
        #计算softmax
        softmax_weight = torch.nn.functional.softmax(torch.tensor(probdiff))
        print(f"Softmax:{softmax_weight}")
        #添加权重
        for history_index, history in enumerate(block["history"]):
            block["history"][history_index].append(probdiff[history_index].item())
            block["history"][history_index].append(probsum[history_index].item())
            block["history"][history_index].append(softmax_weight.tolist()[history_index])

        with open(r"chat_0.8M_encoded.json",'w') as file:
            file.write(json.dumps(data[:block_index],ensure_ascii=False))