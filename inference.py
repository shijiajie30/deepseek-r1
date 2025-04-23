from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 加载微调后的模型和分词器
model_path = './final_saved_model'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 构建推理pipeline
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
)

prompt = 'tell me some singing skills'
generated_text = pipe(prompt, max_length=200, num_return_sequences=1)
print('开始回答：---', generated_text[0]['generated_text'])
