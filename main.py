from transformers import AutoTokenizer, AutoModelForCausalLM
from data_prepare import samples
from datasets import load_dataset
import json

# 加载微调前的模型
print("[1/9] 开始加载预训练模型...")
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ 预训练模型加载完成")

# 第二步，制作数据集
print("\n[2/9] 正在生成训练数据集...")
with open('datasets.jsonl', 'w', encoding='utf-8') as f:
    for s in samples:
        json_line = json.dumps(s, ensure_ascii=False)
        f.write(json_line + '\n')
    else:
        print("✅ 数据集已生成至 datasets.jsonl")

# 第三步，准备训练集和验证集
print("\n[3/9] 正在划分训练集/验证集...")
dataset = load_dataset('json', data_files={'train': 'datasets.jsonl'}, split='train')

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# 第四步，编写tokenizer处理函数
print("\n[4/9] 正在进行文本向量化处理...")
def tokenize_function(examples):
    texts = [f"prompt: {prompt}\ncompletion: {completion}" for prompt, completion in zip(examples['prompt'], examples['completion'])]
    tokens =  tokenizer(texts, padding='max_length', truncation=True, max_length=512)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
print("✅ 文本向量化处理完成")

# 第五步，量化设置
print("\n[5/9] 正在初始化8位量化配置...")
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map='auto')

# 第六步，lora微调设置
print("\n[6/9] 正在配置LoRA微调参数...")
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
   r=8,
   lora_alpha=16,
   lora_dropout=0.5,
   task_type=TaskType.CAUSAL_LM 
)
model = get_peft_model(model, lora_config)

# 第七步，设置训练参数
print("\n[7/9] 正在设置训练超参数...")
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
   output_dir='./finetuned_model',
   num_train_epochs=10,
   per_device_train_batch_size=4,
   gradient_accumulation_steps=8,
   fp16=True,
   logging_steps=10,
   save_steps=100,
   eval_strategy='steps',
   eval_steps=10,
   learning_rate=2e-5,
   logging_dir='./logs',
   run_name='finetuning_qwen1.5b'
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_train_dataset,
  eval_dataset=tokenized_test_dataset
)

# 第八步，开始训练
print("\n[8/9] 🚀 启动模型训练任务...")
trainer.train()
print("✅ 模型训练完成")

# 第九步，保存微调后的模型
print("\n[9/9] 正在保存微调模型...")
save_path = './saved_model'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 释放训练模型内存
import torch
del model, trainer
torch.cuda.empty_cache()

# 保存全量模型（使用量化配置加载基础模型）
final_save_path = './final_saved_model'
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quantization_config,  # 复用量化配置
    device_map='auto'
)
model = PeftModel.from_pretrained(base_model, save_path)
model = model.merge_and_unload()

# 再次释放内存
del base_model
torch.cuda.empty_cache()

model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"✅ 模型保存完成 (LoRA模型: {save_path}, 全量模型: {final_save_path})")