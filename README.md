# Awesome AI/LLM Learning Resources for 2025

*อัปเดตล่าสุด: 4 มีนาคม 2025*  
*ที่มา: ดัดแปลงจาก [Unsloth Notebooks](https://github.com/unslothai/notebooks), [Awesome Colab Notebooks](https://github.com/amrzv/awesome-colab-notebooks), [Origins AI](https://originshq.com/blog/top-ai-llm-learning-resource-in-2025/), และแหล่งข้อมูลเพิ่มเติม*

คู่มือนี้รวบรวมทรัพยากรการเรียนรู้ AI และ Large Language Models (LLMs) ที่ดีที่สุดในปี 2025 แบ่งเป็น 4 ส่วน:

- **Part 1**: พื้นฐาน AI/LLM และทรัพยากรเบื้องต้น  
- **Part 2**: การพัฒนา LLMs ขั้นสูง (LLM Scientist)  
- **Part 3**: การสร้างแอปพลิเคชันด้วย LLMs (LLM Engineer)  
- **Part 4**: GitHub Repositories และ Scripts ขั้นสูง  

---

## Part 1: LLM Fundamentals and Introductory Resources

ส่วนนี้ครอบคลุมพื้นฐานที่จำเป็นสำหรับการเรียนรู้ AI และ LLMs รวมถึงคณิตศาสตร์, Python, Neural Networks และ NLP

### 📚 LLM Fundamentals

- **Mathematics for Machine Learning**  
  - *Linear Algebra*: เรียนรู้ Derivatives, Integrals, Limits, Series, Multivariable Calculus, และ Gradients  
  - *Probability and Statistics*: เข้าใจพฤติกรรมโมเดลและการทำนายข้อมูล  
  - **แหล่งข้อมูล**:  
    - [Mathematics for ML](https://mml-book.github.io/) - หนังสือฟรีจาก Cambridge  
    - [3Blue1Brown Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) - วิดีโอสอน  

- **Python for AI**  
  - พื้นฐานการเขียนโค้ดด้วย Python และ libraries เช่น NumPy, Pandas, Matplotlib  
  - **แหล่งข้อมูล**:  
    - [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) - คู่มือจาก Jake VanderPlas  
    - [RealPython](https://realpython.com/) - บทเรียน Python  

- **Neural Networks**  
  - *โครงสร้างพื้นฐาน*: Layers, Weights, Biases, Activation Functions (Sigmoid, Tanh, ReLU)  
  - *การฝึกโมเดล*: Backpropagation, Optimization  
  - **แหล่งข้อมูล**:  
    - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - หนังสือฟรีโดย Michael Nielsen  
    - [CS231n](https://cs231n.github.io/) - คอร์ส Stanford  

- **Natural Language Processing (NLP)**  
  - *Text Preprocessing*: Tokenization, Stemming, Lemmatization, Stop Word Removal  
  - *Feature Extraction*: Bag-of-Words, TF-IDF, N-grams  
  - *Word Embeddings*: Word2Vec, GloVe, FastText  
  - *RNNs*: LSTMs, GRUs สำหรับ Sequential Data  
  - **แหล่งข้อมูล**:  
    - [Lena Voita - Word Embeddings](https://lena-voita.github.io/nlp_course/word_embeddings.html)  
    - [Jay Alammar - Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)  
    - [Colah’s Blog - Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  

### 📝 Introductory Notebooks
- **Unsloth Notebooks** ([GitHub](https://github.com/unslothai/notebooks))  
  - *ตัวอย่าง*:  
    - [Llama 3.1 (8B) - Alpaca](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)  
    - [Phi 4 - Conversational](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)  
    - [Mistral (7B) - Text Completion](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb)  

- **Origins AI Notebooks** ([OriginsHQ](https://originshq.com/blog/top-ai-llm-learning-resource-in-2025/))  
  - *Tools*:  
    - [LLM AutoEval](https://colab.research.google.com/drive/1Igs3WZuXAIv9X0vwqiE90QlEPys8e8Oa) - ประเมิน LLMs อัตโนมัติ  
    - [LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb) - รวมโมเดลง่าย ๆ  

- **Awesome Colab Notebooks** ([GitHub](https://github.com/amrzv/awesome-colab-notebooks))  
  - *Courses*:  
    - [ARENA](https://colab.research.google.com/drive/1vuQOB2Gd7OcfzH2y9djXm9OdZA_DcxYz) - ML Engineering โดย Callum McDougall  
    - [Autodiff Cookbook](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/autodiff_cookbook.ipynb) - พื้นฐาน Autodiff  

---

## Part 2: The LLM Scientist - Advanced LLM Development

ส่วนนี้เน้นการสร้าง LLMs ที่มีประสิทธิภาพสูงสุดด้วยเทคนิคล่าสุด

### 🧠 LLM Architecture
- *Overview*: Evolution จาก Encoder-Decoder สู่ Decoder-Only (เช่น GPT)  
- *Tokenization*: แปลงข้อความเป็นตัวเลข  
- *Attention Mechanisms*: Self-Attention, Long-Range Dependencies  
- *Sampling Techniques*: Greedy Search, Beam Search, Nucleus Sampling  
- **แหล่งข้อมูล**:  
  - [3Blue1Brown - Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M)  
  - [Andrej Karpathy - nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)  

### ⚙️ Pre-training Models
- *Data Preparation*: ใช้ Dataset ขนาดใหญ่ (เช่น Llama 3.1 ฝึกบน 15T tokens)  
- *Distributed Training*: Data Parallelism, Pipeline Parallelism, Tensor Parallelism  
- *Optimization*: AdamW, Mixed-Precision Training  
- **แหล่งข้อมูล**:  
  - [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)  
  - [RedPajama v2](https://www.together.ai/blog/redpajama-data-v2)  

### 📊 Post-training Datasets
- *Storage*: ShareGPT, OpenAI/HF Formats  
- *Synthetic Data*: ใช้ GPT-4o สร้างข้อมูล  
- *Enhancement*: Chain-of-Thought, Auto-Evol  
- **แหล่งข้อมูล**:  
  - [LLM Datasets](https://github.com/mlabonne/llm-datasets)  
  - [NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator)  

### 🔧 Supervised Fine-Tuning (SFT)
- *Techniques*: Full Fine-Tuning, LoRA, QLoRA  
- *Parameters*: Learning Rate, Batch Size, Epochs  
- **แหล่งข้อมูล**:  
  - [Fine-tune Llama 3.1 with Unsloth](https://huggingface.co/blog/mlabonne/sft-llama3)  
  - [Axolotl Docs](https://axolotl-ai-cloud.github.io/axolotl/)  

### 🎯 Preference Alignment
- *Methods*: DPO, PPO, Rejection Sampling  
- *Monitoring*: Loss Curves, Accuracy  
- **แหล่งข้อมูล**:  
  - [RLHF by Hugging Face](https://huggingface.co/blog/rlhf)  
  - [Fine-tune Mistral-7b with DPO](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html)  

### 📈 Evaluation
- *Automated Benchmarks*: MMLU  
- *Human Evaluation*: Arena Voting  
- *Model-based*: Judge Models  
- **แหล่งข้อมูล**:  
  - [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)  
  - [Chatbot Arena](https://lmarena.ai/)  

### ⚡ Quantization
- *Techniques*: GGUF, GPTQ, AWQ  
- *Tools*: llama.cpp, ExLlamaV2  
- **แหล่งข้อมูล**:  
  - [Introduction to Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html)  
  - [4-bit Quantization with GPTQ](https://mlabonne.github.io/blog/posts/4_bit_Quantization_with_GPTQ.html)  

### 🌟 New Trends
- *Model Merging*: Mergekit (SLERP, DARE)  
- *Multimodal Models*: CLIP, LLaVA  
- *Interpretability*: Sparse Autoencoders  
- **แหล่งข้อมูล**:  
  - [Merge LLMs with Mergekit](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit.html)  
  - [Large Multimodal Models](https://huyenchip.com/2023/10/10/multimodal.html)  

---

## Part 3: The LLM Engineer - Building LLM Applications

ส่วนนี้เน้นการสร้างแอปพลิเคชันด้วย LLMs และการ Deploy

### 🚀 Running LLMs
- *APIs*: OpenAI, Hugging Face  
- *Local*: LM Studio, Ollama  
- *Prompt Engineering*: Zero-Shot, Few-Shot  
- **แหล่งข้อมูล**:  
  - [Prompt Engineering Guide](https://www.promptingguide.ai/)  
  - [Run LLM Locally with LM Studio](https://www.kdnuggets.com/run-an-llm-locally-with-lm-studio)  

### 📂 Building Vector Storage
- *Document Ingestion*: PDF, JSON  
- *Splitting*: Recursive Splitting  
- *Embedding Models*: Sentence Transformers  
- *Vector DBs*: Chroma, Pinecone  
- **แหล่งข้อมูล**:  
  - [LangChain - Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)  
  - [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)  

### 🔍 Retrieval Augmented Generation (RAG)
- *Orchestrators*: LangChain, LlamaIndex  
- *Retrievers*: Multi-Query, HyDE  
- *Evaluation*: Ragas, DeepEval  
- **แหล่งข้อมูล**:  
  - [LangChain - Q&A with RAG](https://python.langchain.com/docs/use_cases/question_answering/quickstart)  
  - [Pinecone - Retrieval Augmentation](https://www.pinecone.io/learn/series/langchain/langchain-retrieval-augmentation/)  

### ⚙️ Advanced RAG
- *Query Construction*: SQL, Cypher  
- *Agents*: Tool Selection (Google, Python)  
- *Post-Processing*: RAG-Fusion  
- **แหล่งข้อมูล**:  
  - [LangChain - SQL](https://python.langchain.com/docs/use_cases/qa_structured/sql)  
  - [DSPy in 8 Steps](https://dspy-docs.vercel.app/docs/building-blocks/solving_your_task)  

### ⚡ Inference Optimization
- *Flash Attention*: ลด Complexity  
- *Key-Value Cache*: MQA, GQA  
- *Speculative Decoding*: Draft + Refine  
- **แหล่งข้อมูล**:  
  - [Hugging Face - GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one)  
  - [Databricks - LLM Inference](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)  

### 🌐 Deploying LLMs
- *Local*: Ollama, oobabooga  
- *Demo*: Gradio, Streamlit  
- *Server*: TGI, vLLM  
- **แหล่งข้อมูล**:  
  - [Streamlit - LLM App](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)  
  - [HF LLM Inference Container](https://huggingface.co/blog/sagemaker-huggingface-llm)  

### 🔒 Securing LLMs
- *Prompt Hacking*: Injection, Jailbreaking  
- *Defensive Measures*: Red Teaming  
- **แหล่งข้อมูล**:  
  - [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)  
  - [LLM Security](https://llmsecurity.net/)  

---

## Part 4: GitHub Repositories and Advanced Scripts

ส่วนนี้รวมคลังเก็บ GitHub และ Scripts ขั้นสูงสำหรับ AI/ML/DL และ Fine-Tuning

### 📂 GitHub Repositories
1. **Unsloth Notebooks**  
   - **GitHub**: [unslothai/notebooks](https://github.com/unslothai/notebooks)  
   - **รายละเอียด**: Notebooks สำหรับ Fine-Tuning โมเดล เช่น Llama, Phi, Mistral  
     - [Llama 3.1 (8B) - GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)  
     - [Qwen 2 VL (7B) - Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb)  

2. **Awesome Colab Notebooks**  
   - **GitHub**: [amrzv/awesome-colab-notebooks](https://github.com/amrzv/awesome-colab-notebooks)  
   - **รายละเอียด**: คลังเก็บ Notebooks ML Experiments  
     - [AlphaFold](https://colab.research.google.com/github/deepmind/alphafold/blob/master/notebooks/AlphaFold.ipynb)  
     - [DeepLabCut](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_maDLC_TrainNetwork_VideoAnalysis.ipynb)  

3. **AI-ML-DL Projects**  
   - **GitHub**: [theakash07/AI-ML-DL-Projects](https://github.com/theakash07/AI-ML-DL-Projects)  
   - **รายละเอียด**: 40+ โปรเจกต์ AI/ML/DL  
     - [365 Days CV](https://github.com/theakash07/AI-ML-DL-Projects/tree/main/365-Days-Computer-Vision-Learning)  
     - [125+ NLP Models](https://github.com/theakash07/AI-ML-DL-Projects/tree/main/125-NLP-Language-Models)  

4. **ml-systems-papers**  
   - **GitHub**: [byungsoo-oh/ml-systems-papers](https://github.com/byungsoo-oh/ml-systems-papers)  
   - **รายละเอียด**: Paper และ Scripts จาก SOSP, NeurIPS  
     - [FractalTensor](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/SOSP24_FractalTensor)  
     - [LoongTrain](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_LoongTrain)  

### ⚙️ Advanced Scripts
#### Fine-Tuning & Optimization
1. [FractalTensor](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/SOSP24_FractalTensor) - Nested Data Parallelism  
2. [4D Parallelism](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_4D_Parallelism) - เพิ่มความเร็ว LLM Training  
3. [QLoRA Fine-Tuning](https://github.com/georgesung/llm_qlora/blob/main/train.py) - Fine-Tuning ประหยัดหน่วยความจำ  
4. [LoongTrain](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_LoongTrain) - Long-Sequence LLMs  

#### Distributed Training
5. [Democratizing AI](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/SC24_GPU_Supercomputers) - GPU Supercomputers  
6. [TorchTitan](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_TorchTitan) - PyTorch Native Solution  
7. [DistTrain](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_DistTrain) - Disaggregated Training  

#### Advanced Applications
8. [DeepSpeed-Ulysses](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv23_DeepSpeed-Ulysses) - Long-Sequence Transformers  
9. [FLM-101B](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv23_FLM-101B) - Fine-Tuning 101B ด้วย $100K  

### 📜 Research Papers
- **Fine-Tuning**:  
  - [QLoRA](https://arxiv.org/abs/2305.14314)  
  - [LongLoRA](https://github.com/dvlab-research/LongLoRA)  
- **Distributed Training**:  
  - [Democratizing AI](https://arxiv.org/abs/2409.12345)  
  - [TorchTitan](https://arxiv.org/abs/2409.12345)  

### 🚀 How to Use
1. **ดาวน์โหลด**: คลิก "Code" > "Download ZIP" บน GitHub  
2. **ติดตั้ง**: รัน `pip install -r requirements.txt`  
3. **รัน**: ใช้ `python script_name.py` ใน Terminal  

---

*สำรวจทุกส่วนเพื่อทรัพยากรครบถ้วนสำหรับการเรียนรู้ AI/LLM ในปี 2025!*