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

## Group Relative Policy Optimization (GRPO)

![GRPO Objective and Training Flow](grpo.png)

GRPO trains a policy by **comparing multiple solutions to the same problem**, rather than scoring each solution in isolation.

For a fixed input (in our case, a TextWorld game prompt), the model generates a **group of complete trajectories**. Each trajectory represents a full attempt at solving the task. After all attempts finish, a single reward is assigned to each trajectory based on how well it performed.

Instead of learning an absolute notion of “good” or “bad,” GRPO asks a simpler question:
**which trajectories in this group were better than the others?**

The rewards are normalized within the group to produce relative advantages. These advantages are then applied to every token in the corresponding trajectory, encouraging the model to increase the probability of actions that appeared in better-than-average solutions and decrease it for worse ones.

A reference policy is used to regularize training, preventing the model from drifting too far from its original behavior and keeping learning stable.
