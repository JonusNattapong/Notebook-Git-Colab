# Fine-Tuning Large Language Models (LLMs) Resources

This document provides a curated list of resources for fine-tuning Large Language Models (LLMs), including links to GitHub repositories, Hugging Face models, and articles. The resources are categorized for easier navigation.

## QLoRA Fine-Tuning

- **QLoRA Fine-Tuning Pipeline**: [GitHub](https://github.com/WeixuanJiang/Qlora-Fine-Tuning-Pipeline) - Scripts and configuration files for fine-tuning LLMs using QLoRA.
- **LLM Fine-Tuning for Programming Queries**: [GitHub](https://github.com/Avani1297/LLM-Fine-Tuning-Project-for-Programming-Queries) - Fine-tuning LLMs on Stack Overflow datasets.
- **Llama-2 Fine-Tuning with QLoRA**: [GitHub](https://github.com/mert-delibalta/llama2-fine-tune-qlora) - Fine-tuning Llama-2 using QLoRA.
- **Llama2 Fine-Tuning with QLoRA (torchtune)**: [PyTorch](https://pytorch.org/torchtune/stable/tutorials/qlora_finetune.html) - Fine-tuning Llama2 using QLoRA with torchtune.
- **Fine-Tuning LLMs using QLoRA**: [GitHub](https://github.com/georgesung/llm_qlora) - Fine-tuning LLMs using QLoRA.

## Other Fine-Tuning Resources

- **FLUX.1 Fine-Tuning**: [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev/discussions/196) - Fine-tuning FLUX.1.
- **BERT Fine-Tuning with NVIDIA NGC**: [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/bert_workshop) - Fine-tuning BERT.
- **Fine-Tuning LLMs with Kiln AI**: [GitHub](https://github.com/Kiln-AI/kiln) - Fine-tuning with Kiln AI, including a UI.
- **Fine-Tuning LLMs with Hugging Face**: [GitHub](https://github.com/Acerkhan/generative-ai-with-MS) - Step-by-step tutorial for fine-tuning with Hugging Face Transformers.
- **Fine-Tuning LLMs with Node-RED Flow**: [GitHub](https://github.com/rozek/node-red-flow-gpt4all-unfiltered) - Fine-tuning GPT4All using Node-RED.
- **Fine-Tuning LLMs with OpenAI**: [GitHub](https://github.com/openai/openai-python/blob/main/examples/fine_tuning.py) - Fine-tuning using the OpenAI API.
- **Fine-Tuning LLMs with Azure OpenAI**: [GitHub](https://github.com/Azure/azure-ai-openai/blob/main/samples/fine_tuning.ipynb) - Fine-tuning using Azure OpenAI Service.
- **Fine-Tuning LLMs with AWS SageMaker**: [GitHub](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/transformers/transformers_fine_tuning.ipynb) - Fine-tuning using AWS SageMaker.
- **Fine-Tuning LLMs with Google AI**: [GitHub](https://github.com/googleapis/python-aiplatform/blob/main/samples/v1beta1/fine_tune_model_sample.py) - Fine-tuning using Google AI Platform.
- **Fine-Tuning LLMs with Microsoft DeepSpeed**: [GitHub](https://github.com/microsoft/DeepSpeed/blob/main/examples/fine_tune.py) - Fine-tuning using DeepSpeed.
- **Fine-Tuning LLMs with NVIDIA Triton**: [GitHub](https://github.com/NVIDIA/triton-inference-server/blob/main/docs/Training.md) - Fine-tuning using NVIDIA Triton.

## Additional Resources (Various Platforms)

The document also includes numerous links to resources for fine-tuning on platforms like Azure, AWS, Hugging Face, and using tools like FastAI, PyTorch Lightning, TensorFlow, Keras, ONNX Runtime, and OpenVINO. These resources cover a wide range of techniques and frameworks for fine-tuning AI models. There are also links to academic papers on arXiv related to advanced fine-tuning techniques and distributed training.

## Kimi Fine-Tuning Resources
 
#### **1. QLoRA Fine-Tuning Pipeline**
- **GitHub**: [WeixuanJiang/Qlora-Fine-Tuning-Pipeline](https://github.com/WeixuanJiang/Qlora-Fine-Tuning-Pipeline)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน QLoRA (Quantized Low-Rank Adaptation) พร้อม Script และ Configuration Files สำหรับการ Fine-Tuning และ Inference.  
    - **Script ตัวอย่าง**:  
      - [Train Script](https://github.com/WeixuanJiang/Qlora-Fine-Tuning-Pipeline/blob/main/scripts/run_training.bat)  
      - [Merge Script](https://github.com/WeixuanJiang/Qlora-Fine-Tuning-Pipeline/blob/main/scripts/run_merge_multiple_loras.bat)  

#### **2. LLM Fine-Tuning for Programming Queries**
- **GitHub**: [Avani1297/LLM-Fine-Tuning-Project-for-Programming-Queries](https://github.com/Avani1297/LLM-Fine-Tuning-Project-for-Programming-Queries)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs บน Stack Overflow datasets ผ่าน Hugging Face และ Vast.ai.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Script](https://github.com/Avani1297/LLM-Fine-Tuning-Project-for-Programming-Queries/blob/main/train.py)  

#### **3. FLUX.1 Fine-Tuning**
- **Hugging Face**: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/discussions/196)  
  - **รายละเอียด**: วิธีการ Fine-Tuning FLUX.1 ผ่าน AI Toolkit.  
    - **Script ตัวอย่าง**:  
      - [Train Script](https://github.com/ostris/ai-toolkit/blob/main/train_lora_flux_24gb.py)  

#### **4. Llama-2 Fine-Tuning with QLoRA**
- **GitHub**: [mert-delibalta/llama2-fine-tune-qlora](https://github.com/mert-delibalta/llama2-fine-tune-qlora)  
  - **รายละเอียด**: วิธีการ Fine-Tuning Llama-2 ผ่าน QLoRA พร้อม Script และ Configuration Files.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Script](https://github.com/mert-delibalta/llama2-fine-tune-qlora/blob/main/train.py)  

#### **5. BERT Fine-Tuning with NVIDIA NGC**
- **NVIDIA NGC**: [Fine-Tune and Optimize BERT](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/bert_workshop)  
  - **รายละเอียด**: วิธีการ Fine-Tuning BERT ผ่าน NVIDIA NGC.  
    - **Script ตัวอย่าง**:  
      - [Training Notebook](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/bert_workshop)  

#### **6. Llama2 Fine-Tuning with QLoRA (torchtune)**
- **PyTorch**: [Fine-Tuning Llama2 with QLoRA](https://pytorch.org/torchtune/stable/tutorials/qlora_finetune.html)  
  - **รายละเอียด**: วิธีการ Fine-Tuning Llama2 ผ่าน QLoRA พร้อม Script และ Configuration Files.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Command](https://pytorch.org/torchtune/stable/tutorials/qlora_finetune.html)

### 5-10 ลิ้งค์ Script สำหรับ Fine-Tuning Uncensored AI Models

#### **1. Fine-Tuning LLMs using QLoRA**
- **GitHub**: [georgesung/llm_qlora](https://github.com/georgesung/llm_qlora)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน QLoRA พร้อม Script และ Configuration Files.  
    - **Script ตัวอย่าง**:  
      - [Train Script](https://github.com/georgesung/llm_qlora/blob/main/train.py)  
      - [Config File](https://github.com/georgesung/llm_qlora/blob/main/configs/llama3_8b_chat_uncensored.yaml) 

#### **2. Fine-Tuning LLMs with Kiln AI**
- **GitHub**: [Kiln-AI/kiln](https://github.com/Kiln-AI/kiln)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน Kiln AI พร้อม UI สำหรับการ Fine-Tuning และ Synthetic Data Generation.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Guide](https://github.com/Kiln-AI/kiln/blob/main/guides/Fine%20Tuning%20LLM%20Models%20Guide.md) 

#### **3. Fine-Tuning LLMs with Hugging Face**
- **GitHub**: [Acerkhan/generative-ai-with-MS](https://github.com/Acerkhan/generative-ai-with-MS/blob/main/18-fine-tuning/README.md)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน Hugging Face Transformers พร้อม Step-by-Step Tutorial.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Script](https://github.com/Acerkhan/generative-ai-with-MS/blob/main/18-fine-tuning/README.md) 

#### **4. Fine-Tuning LLMs with Node-RED Flow**
- **GitHub**: [rozek/node-red-flow-gpt4all-unfiltered](https://github.com/rozek/node-red-flow-gpt4all-unfiltered)  
  - **รายละเอียด**: วิธีการ Fine-Tuning GPT4All ผ่าน Node-RED Flow พร้อม Function Node สำหรับการ Inference.  
    - **Script ตัวอย่าง**:  
      - [Function Node](https://github.com/rozek/node-red-flow-gpt4all-unfiltered/blob/main/GPT4All-unfiltered-Function.json) 

#### **5. Fine-Tuning LLMs with OpenAI**
- **GitHub**: [OpenAI Fine-Tuning](https://github.com/openai/openai-python/blob/main/examples/fine_tuning.py)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน OpenAI API พร้อม Script และ Example.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Script](https://github.com/openai/openai-python/blob/main/examples/fine_tuning.py) 

#### **6. Fine-Tuning LLMs with Azure OpenAI**
- **GitHub**: [Azure OpenAI Fine-Tuning](https://github.com/Azure/azure-ai-openai/blob/main/samples/fine_tuning.ipynb)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน Azure OpenAI Service พร้อม Notebook และ Example.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Notebook](https://github.com/Azure/azure-ai-openai/blob/main/samples/fine_tuning.ipynb) 

#### **7. Fine-Tuning LLMs with AWS SageMaker**
- **GitHub**: [AWS SageMaker Fine-Tuning](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/transformers/transformers_fine_tuning.ipynb)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน AWS SageMaker พร้อม Notebook และ Example.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Notebook](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/transformers/transformers_fine_tuning.ipynb) 

#### **8. Fine-Tuning LLMs with Google AI**
- **GitHub**: [Google AI Fine-Tuning](https://github.com/googleapis/python-aiplatform/blob/main/samples/v1beta1/fine_tune_model_sample.py)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน Google AI Platform พร้อม Script และ Example.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Script](https://github.com/googleapis/python-aiplatform/blob/main/samples/v1beta1/fine_tune_model_sample.py) 

#### **9. Fine-Tuning LLMs with Microsoft DeepSpeed**
- **GitHub**: [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed/blob/main/examples/fine_tune.py)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน Microsoft DeepSpeed พร้อม Script และ Example.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Script](https://github.com/microsoft/DeepSpeed/blob/main/examples/fine_tune.py) 

#### **10. Fine-Tuning LLMs with NVIDIA Triton**
- **GitHub**: [NVIDIA Triton](https://github.com/NVIDIA/triton-inference-server/blob/main/docs/Training.md)  
  - **รายละเอียด**: วิธีการ Fine-Tuning LLMs ผ่าน NVIDIA Triton พร้อม Documentation และ Example.  
    - **Script ตัวอย่าง**:  
      - [Fine-Tuning Documentation](https://github.com/NVIDIA/triton-inference-server/blob/main/docs/Training.md)
