# Robotics and ML-Tutor-LLM

A fine-tuned Llama 3.2 3B chatbot specialized in robotics and machine learning education.

Built using:
- Unsloth + LoRA fine-tuning
- Hugging Face Transformers
- Gradio interface
- VRAM monitoring tools

The project explores lightweight domain-specific LLM fine-tuning for educational tutoring.

The model was trained on a custom educational dataset consisting
of approximately 100 manually curated machine learning and robotics
question-answer pairs. The dataset was manually assembled and refined using AI-assisted
generation workflows to create educational machine learning and
robotics explanations in an instruction-response format.

The dataset emphasized:
- simplified explanations,
- step-by-step reasoning,
- beginner accessibility,
- and example-driven teaching.

![Demo](demo.png)

![Demo](Vram-Usage-Graph.png)

## Features

- Step-by-step ML explanations
- Fine-tuned Llama 3.2 3B model
- Robotics and ML tutoring focus
- LoRA PEFT training pipeline
- Gradio chat interface
- VRAM usage visualization
- Optimized for low-VRAM environments

## Architecture / Workflow

```text
                 ┌──────────────────────┐
                 │   Custom JSON Dataset │
                 │  (Robotics + ML QA)   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Dataset Formatting    │
                 │ format_example()      │
                 │ Instruction → Prompt  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Hugging Face Dataset │
                 │ Dataset.from_list()  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │ Base Model Loading          │
                 │ Llama 3.2 3B (Unsloth)      │
                 │ 4-bit Quantization          │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │ LoRA / PEFT Fine-Tuning     │
                 │ q_proj, v_proj, k_proj,     │
                 │ o_proj adaptation           │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │ SFTTrainer Training         │
                 │ Supervised Fine-Tuning      │
                 │ FP16 + Gradient Accum.      │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │ VRAM Monitoring Callback    │
                 │ CUDA Memory Tracking        │
                 │ Peak VRAM Visualization     │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │ Fine-Tuned Tutor Model      │
                 │ Robotics + ML Assistant     │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │ Inference Pipeline          │
                 │ Prompt Construction         │
                 │ Tokenization + Generation   │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │ Gradio Chat Interface       │
                 │ Interactive Web Chatbot     │
                 └─────────────────────────────┘
```  
## Example Interaction

### User
> What is linear regression in ML?

### Model

> Simple Explanation

>Linear Regression
>Simple Linear Model
>
>Step-by-Step:
>
>1.Calculate slope (w)
>
>2.Calculate intercept(b)
>
>Example: y = w*x + b
>
>Key Idea: Straight Line
>
>Summary:
>
>  - Easy to understand
>  - Predicts well for simple data patterns
>
>Limitations:
>
>  - Not suitable for complex datasets


### User
> What is an actuator in robotics?


### Model

>An ACTUATOR is a device that converts energy into motion. In simple terms, it moves something.
>
>Example: A motor turning a wheel.
>
>Step-by-Step:
>
>1.Energy input (e.g., electricity)
>
>2.Movement output
>
>Key Idea: Motion generation.

## Tech Stack

- Python
- PyTorch
- Unsloth
- Transformers
- PEFT / LoRA
- Gradio
- Matplotlib

## Failure Analysis

The model struggles with:

- overly simplified answers
- ambiguous prompts
- long chain-of-thought reasoning
- highly specialized research terminology

## Example of A Weak Response

### User
>What are support vector machines?

### Model


>Key Idea: Separating Hyperplanes

>Simple Explanation:
>Separate Classes with Lines
>
>Example: Face Recognition
>
>Key Idea: Maximizes Margin
>
>Machine Learning Technique

## Future Improvements

- Better dataset curation
- RAG integration
- Model evaluation benchmarks
- Streaming responses
- Multi-turn memory
- Quantized deployment
