import os
import torch
import argparse
import re
import json
import time
from typing import Optional, List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from peft import LoraConfig, PeftModel
import wandb

class BloodRelationsTrainer:
    """Unified trainer class for both SFT and GRPO training with inference capabilities."""
    
    def __init__(self, args):
        """Initialize the trainer with configuration arguments."""
        self.args = args
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        
        # Inference model cache
        self._inference_model = None
        self._inference_tokenizer = None
        
        # Constants
        # self.reasoning_start = "<reasoning>"
        # self.reasoning_end = "</reasoning>"
        # self.solution_start = "<answer>"
        # self.solution_end = "</answer>"
    
        self.system_prompt_3 = """/no_think You are an expert question generator. Generate multiple-choice questions and respond with valid JSON only.
<think> </think>
Your task: Generate a challenging multiple-choice question from one of these topics:
- Blood Relations
- Seating Arrangements  
- Truth-teller and Liar Problems
'/no_think'
CRITICAL REQUIREMENTS:
1. Output ONLY valid JSON in the exact format below
2. No commentary, explanations, or extra text
3. Question must be ≤150 words
4. Exactly 4 choices labeled A), B), C), D)
5. Answer must be single letter: A, B, C, or D
'/no_think'
EXACT JSON FORMAT REQUIRED:
{
  "questions": [
    {
      "question": "Complete question with background, character statements, and query (≤150 words)",
      "choices": [
        "A) Option 1",
        "B) Option 2", 
        "C) Option 3",
        "D) Option 4"
      ],
      "answer": "A"
    }
  ]
}
'/no_think'
Generate exactly 1 question. Output only the JSON structure above.
Please DO NOT reveal the solution steps or any intermediate reasoning.'/no_think'"""
        # Setup environment
        self._setup_environment()
        
        # Display configuration
        self._display_config()
    
    def _setup_environment(self):
        """Setup environment variables and GPU configuration."""
        # GPU selection - parse gpu_ids from args
        gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(',') if x.strip().isdigit()]
        if not gpu_ids:
            gpu_ids = [0]  # Default to GPU 0 if no valid IDs provided
        
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpu_ids)))
        
        # Add environment variables for distributed training
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "8080")
        
        # ROCm optimization flags for vLLM (if using GRPO)
        if self.args.training_type == 'grpo':
            os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")
            os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
        
        print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}")
    
    def _display_config(self):
        """Display training configuration."""
        print("=" * 60)
        print(f"BLOOD RELATIONS {self.args.training_type.upper()} TRAINER")
        print("=" * 60)
        print(f"Training Type: {self.args.training_type}")
        print(f"Mode: {self.args.mode}")
        print(f"Model: {self.args.model_name}")
        print(f"Output directory: {self.args.output_dir}")
        print(f"Dataset file: {self.args.dataset_file}")
        print(f"Learning rate: {self.args.learning_rate}")
        print(f"Epochs: {self.args.num_train_epochs}")
        print(f"Batch size: {self.args.per_device_train_batch_size}")
        print(f"LoRA rank: {self.args.lora_r}")
        print(f"LoRA alpha: {self.args.lora_alpha}")
        print(f"Max sequence length: {self.args.max_seq_length}")
        print(f"GPU IDs: {self.args.gpu_ids}")
        print("=" * 60)
    
    def load_dataset(self) -> Dataset:
        """Load and process the blood relations dataset."""
        print(f"Loading dataset from: {self.args.dataset_file}")
        
        # Load raw data once
        raw_items = self._load_raw_dataset()
        self.dataset = self._format_generation_dataset(raw_items)
    
    def _load_raw_dataset(self) -> List[Dict[str, str]]:
        """Load raw dataset items from JSON file."""
        items = []
        
        try:
            with open(self.args.dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both array format and single object format
            if isinstance(data, list):
                # Array of question objects
                question_objects = data
            else:
                # Single object, wrap in list
                question_objects = [data]
            
            for idx, question_obj in enumerate(question_objects, 1):
                            try:
                                if "question" in question_obj:
                                    topic = question_obj.get("topic", "blood_relations")
                                    difficulty = question_obj.get("difficulty", "medium")
                                    question_text = question_obj["question"]
                                    choices = question_obj.get("choices", [])
                                    correct_answer = question_obj.get("correct_answer", question_obj.get("answer", "")).strip()
                                    explanation = question_obj.get("explanation", "")
                                    
                                    if question_text and correct_answer:
                                        items.append({
                                            'topic': topic,
                                            'difficulty': difficulty,
                                            'question': question_text,
                                            'choices': choices,
                                            'correct_answer': correct_answer
                                        })
                                        
                                        if idx <= 3:  # Show first 3 for debugging
                                            print(f"✓ Loaded example {idx}: {topic} - {difficulty}")
                                    else:
                                        print(f"Warning: Incomplete question data at index {idx}")
                                        
                            except Exception as e:
                                print(f"Warning: Error processing question at index {idx}: {e}")
                                continue

        except FileNotFoundError:
            print(f"Error: Dataset file {self.args.dataset_file} not found.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {self.args.dataset_file}: {e}")
            raise
        except Exception as e:
            print(f"Error: Unexpected error loading dataset: {e}")
            raise
    
        if not items:
            print("Error: No valid items found in dataset file.")
            raise ValueError("No valid training data found")
        
        print(f"Successfully loaded {len(items)} questions from JSON file")
        return items

    def _format_generation_dataset(self, raw_items: List[Dict[str, str]]) -> Dataset:
        """Format raw items for GRPO question generation training."""
        formatted_items = []
        
        for item in raw_items:
            # Simple generation prompt
            generation_prompt = "Generate a challenging logical reasoning question."
            
            # Target: simplified question structure
            target_question = {
                "question": item['question'],
                "choices": item['choices'],
                "answer": item.get('correct_answer', item.get('answer', 'A'))
            }
            
            # Format for GRPO training
            formatted_items.append({
                'prompt': [
                    # {"role": "system", "content": "<think>\n\n</think>\n"},
                    {'role': 'system', 'content': self.system_prompt_3},
                    {'role': 'user', 'content': generation_prompt}
                ],
                'target_question': target_question,
                'topic': item.get('topic', 'reasoning'),
                'difficulty': item.get('difficulty', 'medium')
            })
        
        print(f"Created {len(formatted_items)} simplified question generation samples")
        return Dataset.from_list(formatted_items)


    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        print(f"Loading model: {self.args.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
                
        # Setup model configuration
        self.model.config.use_cache = False
        if hasattr(self.model.config, 'pretraining_tp'):
            self.model.config.pretraining_tp = 1
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name, 
            trust_remote_code=True, 
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = self.args.max_seq_length
        
        print("Model and tokenizer loaded successfully")
    
    def setup_peft_config(self) -> LoraConfig:
        """Setup LoRA configuration."""
        return LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )
    
    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.args.disable_wandb:
            try:
                wandb.init(
                    project=self.args.wandb_project,
                    name=self.args.wandb_run_name,
                    config={
                        "training_type": self.args.training_type,
                        "model_name": self.args.model_name,
                        "learning_rate": self.args.learning_rate,
                        "num_train_epochs": self.args.num_train_epochs,
                        "per_device_train_batch_size": self.args.per_device_train_batch_size,
                        "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                        "lora_r": self.args.lora_r,
                        "lora_alpha": self.args.lora_alpha,
                        "lora_dropout": self.args.lora_dropout,
                        "max_seq_length": self.args.max_seq_length,
                        "dataset_file": self.args.dataset_file,
                        "gpu_ids": self.args.gpu_ids,
                    }
                )
                print("Weights & Biases initialized successfully")
            except Exception as e:
                print(f"WandB initialization failed: {e}. Training will continue without WandB.")
                self.args.disable_wandb = True
    
    def train_grpo(self):
        """Train using Group Relative Policy Optimization for question generation."""
        print("Starting GRPO training for question generation...")
        
        # Setup training arguments
        training_args = GRPOConfig(
            output_dir=self.args.output_dir,
            learning_rate=self.args.learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_steps=100,
            lr_scheduler_type='cosine_with_restarts',
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_generations=4,
            max_prompt_length=self.args.max_prompt_length,
            max_completion_length=self.args.max_seq_length - self.args.max_prompt_length,
            num_train_epochs=self.args.num_train_epochs,
            save_steps=100,
            log_on_each_node=False,
            use_vllm=True,
            vllm_gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            vllm_mode="colocate",
            report_to="wandb" if not self.args.disable_wandb else "none",
            generation_kwargs={
                "temperature": self.args.temperature,
                "max_tokens": self.args.max_seq_length - self.args.max_prompt_length,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
                # "enable_thinking": False,x
            }
        )
        
        # Setup LoRA config
        peft_config = self.setup_peft_config()
        
        # Create trainer with question generation reward functions
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[self._question_generation_reward_func],
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
        )
        
        # Add callbacks
        self.trainer.add_callback(self._AdjustContextLengthCallback())
        
        # Start training
        self.trainer.train()
        
        print("GRPO question generation training completed successfully")
    
    def _question_generation_reward_func(self, prompts, completions, target_question, topic, difficulty, **kwargs) -> List[float]:
        """Combined reward function for question generation quality."""
        format_scores = self._json_format_reward(completions)
        content_scores = self._content_quality_reward(completions, target_question, topic, difficulty)
        length_scores = self._generation_length_reward(completions)
        
        final_rewards = []
        num_samples = len(completions)

        for i in range(num_samples):
            format_score = format_scores[i]
            content_score = content_scores[i]
            length_score = length_scores[i]
            
            # Weighted combination
            total_reward = (
                0.4 * format_score +     # JSON format correctness
                0.4 * content_score +    # Content quality
                0.2 * length_score       # Appropriate length
            )
            
            final_rewards.append(total_reward)
            
            if i == 0:  # Log first sample for debugging
                print(f"[REWARD] Format: {format_score:.3f}, "
                      f"Content: {content_score:.3f}, "
                      f"Length: {length_score:.3f}, "
                      f"Total: {total_reward:.3f}")
        
        return final_rewards
    
    def _json_format_reward(self, completions) -> List[float]:
        """Reward function for valid JSON format."""
        rewards = []
        
        for completion in completions:
            response = completion[0]["content"]
            print(response)
            try:
                def clean_json_block(text: str) -> str:
                    # Remove any lines that start with ``` (and optional language)
                    lines = text.splitlines()
                    filtered = [line for line in lines if not line.strip().startswith("```")]
                    return "\n".join(filtered)

                # Try to parse as JSON
                # print(type(response))
                cleaned = clean_json_block(response)
                print((cleaned))
                parsed = json.loads(cleaned)            
                print("parsed")
                # Check required fields
                required_fields = ["topic", "difficulty", "question", "choices", "correct_answer"]
                if all(field in parsed for field in required_fields):
                    # Check choices format
                    choices = parsed.get("choices", [])
                    if isinstance(choices, list) and len(choices) == 4:
                        # Check if choices follow A), B), C), D) format
                        choice_pattern = re.compile(r'^[ABCD]\)')
                        if all(choice_pattern.match(choice.strip()) for choice in choices):
                            rewards.append(1.0)  # Perfect format
                        else:
                            rewards.append(0.5)  # Good but imperfect format
                    else:
                        rewards.append(0.2)  # Valid JSON but wrong choices
                else:
                    rewards.append(0.1)  # Valid JSON but missing fields
                    
            except json.JSONDecodeError:
                rewards.append(-1.0)  # Invalid JSON
        
        return rewards
        
    def _content_quality_reward(self, completions, target_question, topic, difficulty) -> List[float]:
        """Reward function for question content quality."""
        rewards = []
        
        for completion in completions:
            response = completion[0]["content"]
            
            try:
                parsed = json.loads(response)
                score = 0.0
                
                # Topic matching (0.3 weight)
                if parsed.get("topic") == topic[0]:  # topic is a list in GRPO
                    score += 0.3
                
                # Difficulty matching (0.2 weight)
                if parsed.get("difficulty") == difficulty[0]:  # difficulty is a list in GRPO
                    score += 0.2
                
                # Question quality heuristics (0.3 weight)
                question_text = parsed.get("question", "")
                if len(question_text) > 30:  # Reasonable length
                    score += 0.1
                if "?" in question_text:  # Has question mark
                    score += 0.05
                if len(question_text.split()) > 10:  # Sufficient detail
                    score += 0.1
                if any(keyword in question_text.lower() for keyword in ["how", "what", "who", "which", "where"]):
                    score += 0.05
                
                # Choices quality (0.2 weight)
                choices = parsed.get("choices", [])
                if len(choices) == 4:
                    score += 0.1
                    # Check for variety in choice lengths (avoid all same length)
                    choice_lengths = [len(choice) for choice in choices]
                    if len(set(choice_lengths)) > 2:
                        score += 0.1
                
                rewards.append(score)
                
            except (json.JSONDecodeError, KeyError):
                rewards.append(0.0)  # No content reward for invalid JSON
        
        return rewards
    
    def _generation_length_reward(self, completions) -> List[float]:
        """Reward function for appropriate generation length."""
        rewards = []
        
        for completion in completions:
            response = completion[0]['content']
            response_length = len(response)
            
            # Optimal range for question generation (JSON format)
            optimal_min = 200  # Minimum for complete question
            optimal_max = 800  # Maximum reasonable length
            
            if optimal_min <= response_length <= optimal_max:
                reward = 1.0
            elif response_length < optimal_min:
                # Too short penalty
                reward = response_length / optimal_min
            else:
                # Too long penalty  
                excess = response_length - optimal_max
                reward = max(0.0, 1.0 - (excess / 1000))
            
            rewards.append(reward)
        
        return rewards
    
    class _AdjustContextLengthCallback(TrainerCallback):
        """Callback to adjust context length during training."""
        
        def on_step_begin(self, args, state, control, **kwargs):
            step = state.global_step
            if step >= 1000:
                args.max_prompt_length = 384
            elif step >= 500:
                args.max_completion_length = 512  # Increased for question generation
            
            if step in [500, 1000]:
                print(f"Adjusted context length at step {step}")
    
    def train(self):
        """Main training function that handles both SFT and GRPO."""
        print(f"Starting {self.args.training_type.upper()} training...")
        
        # Load dataset
        self.load_dataset()
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Setup wandb
        self.setup_wandb()

        # train grpo
        self.train_grpo()
        
        # Cleanup
        if not self.args.disable_wandb and wandb.run:
            wandb.finish()
        
        print(f"{self.args.training_type.upper()} training completed!")
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the output directory."""
        if not os.path.exists(self.args.output_dir):
            print(f"Output directory {self.args.output_dir} does not exist")
            return None
        
        checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
        checkpoints = []
        
        for item in os.listdir(self.args.output_dir):
            item_path = os.path.join(self.args.output_dir, item)
            if os.path.isdir(item_path):
                match = checkpoint_pattern.match(item)
                if match:
                    step_num = int(match.group(1))
                    checkpoints.append((step_num, item_path))
        
        if not checkpoints:
            print(f"No checkpoints found in {self.args.output_dir}")
            return None
        
        checkpoints.sort(key=lambda x: x[0])
        latest_checkpoint = checkpoints[-1][1]
        
        print(f"Found {len(checkpoints)} checkpoints. Using latest: {latest_checkpoint}")
        return latest_checkpoint
    
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Blood Relations Trainer")
    
    # General arguments
    parser.add_argument("--model_name", type=str, default="/jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/demo/sft", help="Output directory for checkpoints and model")
    parser.add_argument("--dataset_file", type=str, default="/jupyter-tutorial/Dataset_generation/data_emnlp_final/data_06b8f2a1/family_reasoning_data_entire.json", help="Path to the dataset file (JSON)")
    parser.add_argument("--test_file", type=str, default="test_questions_array.json", help="Path to the test file for inference")
    parser.add_argument("--test_question", type=str, help="Single question for testing inference")
    parser.add_argument("--inference_output", type=str, default="inference.md", help="Output file for inference results")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use (comma-separated)")
    
    # Training arguments
    parser.add_argument("--training_type", type=str, choices=["sft", "grpo"], default="grpo", help="Type of training")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "both"], default="train", help="Mode of operation")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length for GRPO")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for response generation")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    
    # WandB arguments
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="blood_relations", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, help="WandB run name")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Auto-generate wandb run name if not provided
    model_display_name = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{model_display_name}-{args.training_type}-r{args.lora_r}-lr{args.learning_rate}"
    
    # Create trainer
    trainer = BloodRelationsTrainer(args)

    trainer.train()


if __name__ == "__main__":
    """
    BLOOD RELATIONS UNIFIED TRAINER
    ================================

    This script provides a unified interface for training blood relations reasoning models
    using either Supervised Fine-Tuning (SFT) or Group Relative Policy Optimization (GRPO).
    It also includes comprehensive inference capabilities.

    USAGE EXAMPLES:
    1. SFT TRAINING:

       python -m trainer \
           --training_type sft \
           --mode train \
           --model_name /jupyter-tutorial/hf_models/Qwen3-4B \
           --dataset_file /jupyter-tutorial/AAIPL_129_212_191_47/dataset/llama_70b.json \
           --output_dir /jupyter-tutorial/AAIPL_129_212_191_47/ckpt/question_model \
           --learning_rate 2e-5 \
           --num_train_epochs 5 \
           --per_device_train_batch_size 16 \
           --lora_r 32 \
           --lora_alpha 64

   

    2. GRPO TRAINING:

       python -m trainer1 \
           --training_type grpo \
           --mode train \
           --model_name "/jupyter-tutorial/hf_models/Qwen3-4B" \
           --output_dir /jupyter-tutorial/AAIPL_129_212_191_47/ckpt/GRPO_question_model \
           --dataset_file /jupyter-tutorial/Dataset_generation/grpo_final_data_array.json \
           --learning_rate 1e-5 \
           --num_train_epochs 2 \
           --per_device_train_batch_size 4 \
           --gradient_accumulation_steps 2 \
           --lora_r 16 \
           --lora_alpha 32 \
           --vllm_gpu_memory_utilization 0.7

   

    3. INFERENCE ONLY:

       python -m trainer \
           --mode inference \
           --model_name /jupyter-tutorial/AAIPL_129_212_191_47/ckpt/direct_relation_sft_finetuning/checkpoint-20 \
           --output_dir /jupyter-tutorial/inference_output \
           --test_question "If A is B's father and B is C's mother, what is A to C?"

   

    4. TRAIN + INFERENCE:

       python -m trainer \
           --training_type sft \
           --mode both \
           --model_name /jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct \
           --output_dir checkpoints/demo/sft \
           --test_question "If A is B's father and B is C's mother, what is A to C?"

   

    5. BATCH INFERENCE:

       python -m trainer \
           --mode inference \
           --model_name /jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct \
           --output_dir checkpoints/demo/sft \
           --test_file test_questions.txt \
           --inference_output batch_results.md

   

    6. MULTI-GPU TRAINING:

       python -m trainer \
           --training_type sft \
           --mode train \
           --model_name /jupyter-tutorial/hf_models/Llama-3.1-8B-Instruct \
           --gpu_ids "0,1,2,3" \
           --per_device_train_batch_size 2 \
           --gradient_accumulation_steps 4

   

    7. DISABLE WANDB LOGGING:

       python -m trainer \
           --training_type sft \
           --mode train \
           --disable_wandb

   

    KEY FEATURES:
    - Unified SFT and GRPO training in a single file
    - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    - Built-in inference capabilities with model card-like interface
    - Comprehensive reward functions for GRPO training
    - WandB integration for experiment tracking
    - Multi-GPU support
    - Batch inference with accuracy evaluation
    - Structured XML output format for reasoning and answers

    GRPO SPECIFIC FEATURES:
    - Custom reward functions for format, correctness, and length
    - Dynamic context length adjustment during training
    - vLLM integration for efficient generation
    - Support for multiple generations per prompt
   
    INFERENCE FEATURES:
    - Single question inference for quick testing
    - Batch inference for evaluation on test sets
    - Automatic checkpoint detection and loading
    - Structured response extraction and evaluation
    - Markdown output for easy result review
    """
    main()