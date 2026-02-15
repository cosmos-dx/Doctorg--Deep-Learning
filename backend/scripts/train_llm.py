"""
Fine-tune medical LLM using LoRA for efficient training.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalLLMTrainer:
    """Trainer for medical LLM with LoRA fine-tuning."""
    
    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-v0.1",
        output_dir: str = "backend/models/doctorg-medical-llm"
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Base model: {base_model}")
    
    def setup_lora_config(self) -> LoraConfig:
        """Configure LoRA parameters for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            bias="none",
            inference_mode=False
        )
        
        logger.info("LoRA configuration created")
        return lora_config
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization."""
        logger.info(f"Loading model: {self.base_model}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        model = prepare_model_for_kbit_training(model)
        
        logger.info("Model loaded successfully")
        return model, tokenizer
    
    def prepare_dataset(self, train_path: str, val_path: str, tokenizer):
        """Load and prepare training dataset."""
        logger.info("Loading training data")
        
        train_dataset = load_dataset('json', data_files=train_path, split='train')
        val_dataset = load_dataset('json', data_files=val_path, split='train')
        
        def preprocess_function(examples):
            prompts = [
                f"{inst}\n\n{out}" 
                for inst, out in zip(examples['instruction'], examples['output'])
            ]
            
            tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(
        self,
        train_data_path: str = "backend/data/training/train.jsonl",
        val_data_path: str = "backend/data/training/val.jsonl",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ):
        """Execute fine-tuning with LoRA."""
        logger.info("Starting fine-tuning process")
        
        model, tokenizer = self.load_model_and_tokenizer()
        
        lora_config = self.setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        model.print_trainable_parameters()
        
        train_dataset, val_dataset = self.prepare_dataset(
            train_data_path,
            val_data_path,
            tokenizer
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            save_steps=500,
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=3,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            report_to="none"
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        logger.info(f"Saving model to {self.output_dir}")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training completed successfully!")
    
    def test_inference(self, prompt: str):
        """Test the fine-tuned model."""
        logger.info("Testing inference")
        
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        model = AutoModelForCausalLM.from_pretrained(
            self.output_dir,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated response:\n{response}")
        
        return response


def main():
    """Main execution function."""
    trainer = MedicalLLMTrainer(
        base_model="mistralai/Mistral-7B-v0.1",
        output_dir="backend/models/doctorg-medical-llm"
    )
    
    trainer.train(
        train_data_path="backend/data/training/train.jsonl",
        val_data_path="backend/data/training/val.jsonl",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4
    )
    
    test_prompt = """You are a medical AI assistant. Analyze the symptoms and provide a structured medical assessment.

Symptoms: headache, fever, fatigue

Provide your response in the following JSON format:"""
    
    trainer.test_inference(test_prompt)


if __name__ == "__main__":
    main()
