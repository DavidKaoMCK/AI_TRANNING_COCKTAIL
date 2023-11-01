from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# 載入預處理過的文本數據
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path= 'C:\Davidlocal\PMD_AI_TRANNING2\exampleconversation2.txt',
    block_size=128
)

# 為訓練準備數據
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 定義訓練參數
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=5000,
    save_total_limit=2,
)

# 定義 Trainer 對象
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 訓練模型
trainer.train()


prompt = "Give me a recipe of Espresso Martini and tell me how to make it."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=500, do_sample=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)