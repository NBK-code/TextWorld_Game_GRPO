# TextWorld Agent Training with GRPO

This project trains a large language model to solve **TextWorld** interactive fiction games using **Group Relative Policy Optimization (GRPO)**.

We treat **entire game episodes as rollouts** and optimize the model by comparing multiple trajectories for the same game.

## Model
- **LLaMA 3.1 8B Instruct**
- LoRA adapters for parameter-efficient fine-tuning
- Optional 4-bit quantization for memory efficiency

## Core Idea
1. Fix a TextWorld game (level + seed)
2. Generate multiple full episodes (rollouts)
3. Assign a final reward per episode
4. Compute group-relative advantages
5. Update the policy using GRPO on token-level log-probabilities

## Project Structure
- **Notebook 1**: Run TextWorld episodes and generate GRPO training data
- **Notebook 2**: Train the model using GRPO loss

## What is GRPO?

![GRPO Objective and Training Flow](grpo.png)
