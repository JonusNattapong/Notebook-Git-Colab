#  AI/LLM Learning Resources 2025 by zombitx64

*อัปเดตล่าสุด: 4 มีนาคม 2025*

*ที่มา: ดัดแปลงจาก [Unsloth Notebooks](https://github.com/unslothai/notebooks), [Awesome Colab Notebooks](https://github.com/amrzv/awesome-colab-notebooks), [Origins AI](https://originshq.com/blog/top-ai-llm-learning-resource-in-2025/), และแหล่งข้อมูลเพิ่มเติม*

คลังรวมทรัพยากรสำหรับการเรียนรู้ด้าน Artificial Intelligence (AI) และ Large Language Models (LLMs) ครอบคลุมตั้งแต่ความรู้พื้นฐานไปจนถึงการประยุกต์ใช้งานขั้นสูง

## สารบัญ

1. [พื้นฐานที่จำเป็น](#พื้นฐานที่จำเป็น)
   - คณิตศาสตร์สำหรับ Machine Learning
   - Python สำหรับ AI
   - Neural Networks
   - Natural Language Processing (NLP)
2. [แหล่งเรียนรู้เบื้องต้น](#แหล่งเรียนรู้เบื้องต้น)
   - Notebooks แนะนำ
   - คอร์สออนไลน์และบทเรียน
3. [เครื่องมือและทรัพยากร](#เครื่องมือและทรัพยากร)
   - Datasets
   - Tools
   - แหล่งข้อมูลเพิ่มเติม

## พื้นฐานที่จำเป็น

ส่วนนี้ครอบคลุมความรู้พื้นฐานที่จำเป็นสำหรับการพัฒนา AI และ Large Language Models (LLMs) เหมาะสำหรับผู้เริ่มต้นที่ต้องการสร้างรากฐานที่แข็งแกร่งก่อนศึกษาเทคนิคขั้นสูง

---

### 📚 LLM Fundamentals

#### 1. Mathematics for Machine Learning
คณิตศาสตร์เป็นรากฐานสำคัญของ AI และ Machine Learning (ML) โดยเฉพาะสำหรับการทำความเข้าใจอัลกอริทึมและการฝึกโมเดล  
- **Linear Algebra**:  
  - ครอบคลุม Vectors, Matrices, Derivatives, Integrals, Limits, Series, Multivariable Calculus และ Gradient Concepts  
  - จำเป็นสำหรับ Deep Learning และการคำนวณใน Neural Networks  
- **Probability and Statistics**:  
  - เข้าใจพฤติกรรมของโมเดล การกระจายข้อมูล และการทำนาย  
  - หัวข้อสำคัญ เช่น Probability Distributions, Hypothesis Testing, Bayesian Inference  
- **แหล่งข้อมูล**:  
  - [Mathematics for Machine Learning](https://mml-book.github.io/) - หนังสือฟรีจาก Cambridge University Press  
  - [3Blue1Brown - Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) - วิดีโอสอนภาพเคลื่อนไหวเข้าใจง่าย  
  - [Khan Academy - Probability and Statistics](https://www.khanacademy.org/math/statistics-probability) - คอร์สออนไลน์ฟรี  

#### 2. Python for AI
Python เป็นภาษาหลักสำหรับการพัฒนา AI และ LLMs เนื่องจากมี Libraries ที่ทรงพลังและใช้งานง่าย  
- **พื้นฐาน Python**: Variables, Functions, Loops, Data Structures (Lists, Dictionaries)  
- **Libraries สำคัญ**:  
  - *NumPy*: การคำนวณเชิงตัวเลข  
  - *Pandas*: การจัดการข้อมูล  
  - *Matplotlib*: การแสดงผลข้อมูล  
- **แหล่งข้อมูล**:  
  - [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) - คู่มือโดย Jake VanderPlas พร้อม Notebooks  
  - [RealPython](https://realpython.com/) - บทเรียน Python พร้อมตัวอย่างโค้ด  
  - [CS231n Python Tutorial](https://cs231n.github.io/python-numpy-tutorial/) - จาก Stanford  

#### 3. Neural Networks
Neural Networks เป็นหัวใจของ Deep Learning และ LLMs เข้าใจพื้นฐานเพื่อต่อยอดไปสู่ Transformer Models  
- **โครงสร้างพื้นฐาน**:  
  - Layers, Weights, Biases, Activation Functions (Sigmoid, Tanh, ReLU)  
  - การทำงานของ Feedforward และ Backpropagation  
- **การฝึกโมเดล**:  
  - Loss Functions, Optimization (Gradient Descent), Overfitting Prevention  
- **แหล่งข้อมูล**:  
  - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - หนังสือฟรีโดย Michael Nielsen  
  - [CS231n: Convolutional Neural Networks](https://cs231n.github.io/) - คอร์สจาก Stanford  
  - [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - โดย Andrew NG บน Coursera  

#### 4. Natural Language Processing (NLP)
NLP เป็นสาขาที่เชื่อมโยงภาษามนุษย์กับการประมวลผลของเครื่องจักร เป็นพื้นฐานของ LLMs  
- **Text Preprocessing**:  
  - *Tokenization*: แบ่งข้อความเป็นคำหรือประโยค  
  - *Stemming/Lemmatization*: ลดรูปคำให้เหลือราก  
  - *Stop Word Removal*: ลบคำที่ไม่สำคัญ  
- **Feature Extraction Techniques**:  
  - *Bag-of-Words (BoW)*: แปลงข้อความเป็นเวกเตอร์คำ  
  - *TF-IDF*: น้ำหนักคำตามความสำคัญ  
  - *N-grams*: ลำดับคำต่อเนื่อง  
- **Word Embeddings**:  
  - *Word2Vec, GloVe, FastText*: แปลงคำเป็นเวกเตอร์ที่มีความหมายใกล้เคียงกัน  
- **Recurrent Neural Networks (RNNs)**:  
  - ออกแบบสำหรับข้อมูลลำดับ (Sequential Data)  
  - Variants เช่น LSTMs และ GRUs เพื่อจับ Long-Term Dependencies  
- **แหล่งข้อมูล**:  
  - [Lena Voita - Word Embeddings](https://lena-voita.github.io/nlp_course/word_embeddings.html) - อธิบาย Word Embeddings  
  - [RealPython - NLP with spaCy](https://realpython.com/natural-language-processing-spacy-python/) - การใช้ spaCy  
  - [Jay Alammar - Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - ภาพประกอบเข้าใจง่าย  
  - [Colah’s Blog - Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - อธิบาย LSTMs  
  - [Kaggle - NLP Guide](https://www.kaggle.com/learn-guide/natural-language-processing) - คู่มือจาก Kaggle  

---

### 📝 Introductory Notebooks
Notebooks ช่วยให้ผู้เริ่มต้นฝึกฝนทักษะผ่านการรันโค้ดจริงบนแพลตฟอร์ม เช่น Google Colab และ Kaggle

#### 1. Unsloth Notebooks
- **ที่มา**: [GitHub: unslothai/notebooks](https://github.com/unslothai/notebooks)  
- **รายละเอียด**: รวม Notebooks สำหรับ Fine-Tuning และ Inference โมเดล LLMs บน Google Colab และ Kaggle  
- **ตัวอย่าง**:  
  - **GRPO Notebooks**:  
    - [Phi 4 (14B) - GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb)  
    - [Llama 3.1 (8B) - GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)  
  - **Llama Notebooks**:  
    - [Llama 3.2 (1B and 3B) - Conversational](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)  
    - [Llama 3.2 (11B) - Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)  
  - **Mistral Notebooks**:  
    - [Mistral Small (22B) - Alpaca](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Small_(22B)-Alpaca.ipynb)  
    - [Mistral (7B) - Text Completion](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb)  
  - **Kaggle Variants**: มีเวอร์ชันสำหรับ Kaggle เช่น [Kaggle-Llama3.1_(8B)-Alpaca](https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Llama3.1_(8B)-Alpaca.ipynb)  

#### 2. Origins AI Notebooks
- **ที่มา**: [OriginsHQ](https://originshq.com/blog/top-ai-llm-learning-resource-in-2025/)  
- **รายละเอียด**: Notebooks และบทความสำหรับการเรียนรู้ LLMs  
- **Tools**:  
  - [🧐 LLM AutoEval](https://colab.research.google.com/drive/1Igs3WZuXAIv9X0vwqiE90QlEPys8e8Oa) - ประเมิน LLMs อัตโนมัติด้วย RunPod  
  - [🥱 LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb) - รวมโมเดลด้วย MergeKit  
  - [🦎 LazyAxolotl](https://colab.research.google.com/drive/1TsDKNo2riwVmU55gjuBgB1AXVtRRfRHW) - Fine-Tune โมเดลใน Cloud  
  - [⚡ AutoQuant](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4) - Quantize LLMs เป็น GGUF, GPTQ  
- **Fine-Tuning**:  
  - [Fine-tune Llama 3.1 with Unsloth](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z) - บทความ: [Link](https://originshq.com/blog/fine-tune-llama-3-1-ultra-efficiently-with-unsloth/)  
  - [Fine-tune Mistral-7b with QLoRA](https://colab.research.google.com/drive/1o_w0KastmEJNVwT5GoqMCciH-18ca5WS) - ใช้ TRL  

#### 3. Awesome Colab Notebooks
- **ที่มา**: [GitHub: amrzv/awesome-colab-notebooks](https://github.com/amrzv/awesome-colab-notebooks)  
- **รายละเอียด**: คลังเก็บ Notebooks สำหรับ ML Experiments  
- **Courses**:  
  - [ARENA](https://colab.research.google.com/drive/1vuQOB2Gd7OcfzH2y9djXm9OdZA_DcxYz) - ML Engineering โดย Callum McDougall  
  - [Autodiff Cookbook](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/autodiff_cookbook.ipynb) - พื้นฐาน Autodifferentiation  
  - [Machine Learning Simplified](https://colab.research.google.com/github/5x12/themlsbook/blob/master/chapter2/knn.ipynb) - โดย Andrew Wolf  
  - [Deep RL Course](https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/notebooks/unit1/unit1.ipynb) - จาก Hugging Face  
- **Research**:  
  - [AlphaFold](https://colab.research.google.com/github/deepmind/alphafold/blob/master/notebooks/AlphaFold.ipynb) - Protein Structure Prediction  

---

### 🎓 Online Courses and Tutorials
คอร์สออนไลน์และ Tutorials ที่เหมาะสำหรับผู้เริ่มต้น  
- **Andrew NG Machine Learning Course** ([Coursera](https://www.coursera.org/learn/machine-learning)) - พื้นฐาน ML  
- **Deep Learning Specialization** ([Coursera](https://www.coursera.org/specializations/deep-learning)) - 5 คอร์สจาก Andrew NG  
- **NYU-DLSP20** ([GitHub](https://github.com/Atcold/NYU-DLSP20)) - Deep Learning โดย Yann LeCun  
- **mlcourse.ai** ([GitHub](https://github.com/Yorko/mlcourse.ai)) - Open ML Course โดย Yury Kashnitsky  

---

### 📦 Datasets and Tools
- **Kaggle Datasets** ([Kaggle](https://www.kaggle.com/datasets)) - คลัง Dataset สำหรับฝึกฝน  
- **NLP Datasets** ([GitHub](https://github.com/awwsmm/nlp-datasets)) - 100+ ชุดข้อมูล NLP  
- **Hugging Face Transformers** ([GitHub](https://github.com/huggingface/transformers)) - Library สำหรับ Fine-Tuning BERT, GPT  

---

### 🎙️ Additional Learning Resources
- **Podcasts**:  
  - [Lex Fridman Podcast](https://lexfridman.com/podcast/) - สัมภาษณ์ผู้เชี่ยวชาญ AI  
  - [Data Skeptic](https://dataskeptic.com/) - Data Science และ ML  
- **YouTube Channels**:  
  - [Sentdex](https://www.youtube.com/@sentdex) - Tutorials ML  
  - [Corey Schafer](https://www.youtube.com/@coreyschafer) - Python Coding  
- **Communities**:  
  - [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) - ชุมชน ML บน Reddit  
  - [Discord - Unsloth](https://discord.gg/unsloth) - ชุมชนสำหรับถาม-ตอบ  

---

### 🚀 How to Get Started
1. **เลือกหัวข้อ**: เริ่มจาก Mathematics หรือ Python ตามความถนัด  
2. **ฝึกด้วย Notebooks**: รันโค้ดใน Colab เพื่อเรียนรู้ผ่านการปฏิบัติ  
3. **เข้าร่วมคอร์ส**: ลงทะเบียนคอร์สฟรี เช่น Andrew NG เพื่อโครงสร้างการเรียนรู้  

# Awesome AI/LLM Learning Resources for 2025 (Part 2/4)

*อัปเดตล่าสุด: 4 มีนาคม 2025*  
*ที่มา: ดัดแปลงจาก [Unsloth Notebooks](https://github.com/unslothai/notebooks), [Awesome Colab Notebooks](https://github.com/amrzv/awesome-colab-notebooks), [Origins AI](https://originshq.com/blog/top-ai-llm-learning-resource-in-2025/), และแหล่งข้อมูลเพิ่มเติม*

## Part 2: The LLM Scientist - Advanced LLM Development

ส่วนนี้มุ่งเน้นการสร้าง LLMs ที่มีประสิทธิภาพสูงสุดด้วยเทคนิคล่าสุด เหมาะสำหรับนักวิจัยและผู้ที่ต้องการพัฒนาโมเดลตั้งแต่ต้นจนถึงขั้นสูง ครอบคลุมตั้งแต่สถาปัตยกรรม, การ Pre-training, Post-training, Fine-Tuning, Preference Alignment, การประเมินผล, Quantization และเทรนด์ใหม่ ๆ

---

### 🧠 1. LLM Architecture
เข้าใจโครงสร้างพื้นฐานของ LLMs เพื่อออกแบบและปรับแต่งโมเดล  
- **Architectural Overview**:  
  - Evolution จาก *Encoder-Decoder Transformers* (เช่น BERT) สู่ *Decoder-Only* (เช่น GPT)  
  - เรียนรู้การประมวลผลข้อความและการสร้างข้อความในระดับสูง  
- **Tokenization**:  
  - แปลงข้อความเป็นตัวเลข (Numerical Tokens)  
  - เปรียบเทียบวิธี เช่น Byte-Pair Encoding (BPE), WordPiece  
- **Attention Mechanisms**:  
  - *Self-Attention*: จับความสัมพันธ์ในข้อความ  
  - Variants เช่น Multi-Head Attention, Long-Range Dependencies  
- **Sampling Techniques**:  
  - *Deterministic*: Greedy Search, Beam Search  
  - *Probabilistic*: Temperature Sampling, Nucleus Sampling  
- **แหล่งข้อมูล**:  
  - [3Blue1Brown - Visual Intro to Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M) - อธิบาย Transformer ด้วยภาพ  
  - [Andrej Karpathy - nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - สร้าง GPT ขนาดเล็ก (มีวิดีโอ Tokenization: [Link](https://www.youtube.com/watch?v=zduSFxRajkE))  
  - [Lilian Weng - Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - กลไก Attention  
  - [Maxime Labonne - Decoding Strategies](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html) - วิธีการสร้างข้อความ  

---

### ⚙️ 2. Pre-training Models
Pre-training เป็นกระบวนการฝึกโมเดลตั้งแต่เริ่มต้นด้วยข้อมูลขนาดใหญ่ ซึ่งใช้ทรัพยากรสูง แต่จำเป็นสำหรับโมเดลพื้นฐาน  
- **Data Preparation**:  
  - ใช้ Dataset ขนาดใหญ่ (เช่น Llama 3.1 ฝึกบน 15T Tokens)  
  - ขั้นตอน: Curate, Clean, Deduplicate, Tokenize, Quality Filtering  
- **Distributed Training**:  
  - *Data Parallelism*: แบ่ง Batch ไปยัง GPU หลายตัว  
  - *Pipeline Parallelism*: แบ่ง Layers  
  - *Tensor Parallelism*: แยกการคำนวณ  
- **Training Optimization**:  
  - Adaptive Learning Rates, Gradient Clipping, Mixed-Precision Training  
  - Optimizers: AdamW, Lion  
- **Monitoring**:  
  - ติดตาม Loss, Gradients, GPU Usage ด้วย Dashboards  
- **แหล่งข้อมูล**:  
  - [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) - Dataset คุณภาพสูงโดย Penedo et al.  
  - [RedPajama v2](https://www.together.ai/blog/redpajama-data-v2) - Dataset เปิดโดย Weber et al.  
  - [Nanotron](https://github.com/huggingface/nanotron) - ใช้ฝึก SmolLM2 ([Link](https://github.com/huggingface/smollm))  
  - [Parallel Training](https://www.andrew.cmu.edu/course/11-667/lectures/W10L2%20Scaling%20Up%20Parallel%20Training.pdf) - โดย Chenyan Xiong  
  - [Distributed Training](https://arxiv.org/abs/2407.20018) - Paper โดย Duan et al.  
  - [FractalTensor](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/SOSP24_FractalTensor) - Nested Data Parallelism  

---

### 📊 3. Post-training Datasets
Post-training ปรับโมเดลให้ตอบสนองต่อคำสั่งและการสนทนาได้ดีขึ้น  
- **Storage & Chat Templates**:  
  - Formats: ShareGPT, OpenAI/HF  
  - Chat Templates: ChatML, Alpaca  
- **Synthetic Data Generation**:  
  - ใช้โมเดลขั้นสูง (เช่น GPT-4o) สร้างคู่ Instruction-Response  
  - เทคนิค: Diverse Seed Tasks, Effective Prompts  
- **Data Enhancement**:  
  - Verified Outputs (Unit Tests), Rejection Sampling, Auto-Evol ([Paper](https://arxiv.org/abs/2406.00770))  
  - Chain-of-Thought, Branch-Solve-Merge, Persona-based  
- **Quality Filtering**:  
  - Rule-based, Duplicate Removal (MinHash/Embeddings), N-gram Decontamination  
  - ใช้ Reward Models และ Judge LLMs  
- **แหล่งข้อมูล**:  
  - [LLM Datasets](https://github.com/mlabonne/llm-datasets) - คลัง Dataset โดย Maxime Labonne  
  - [Synthetic Data Generator](https://huggingface.co/spaces/argilla/synthetic-data-generator) - โดย Argilla  
  - [NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) - เครื่องมือจัดการ Dataset  
  - [Distilabel](https://distilabel.argilla.io/dev/sections/pipeline_samples/) - สร้างข้อมูลคุณภาพ  
  - [Chat Template](https://huggingface.co/docs/transformers/main/en/chat_templating) - คู่มือจาก Hugging Face  

---

### 🔧 4. Supervised Fine-Tuning (SFT)
SFT ปรับโมเดลให้เป็นผู้ช่วยที่ตอบคำสั่งได้ดี  
- **Training Techniques**:  
  - *Full Fine-Tuning*: อัปเดตทุกพารามิเตอร์ (ใช้ทรัพยากรสูง)  
  - *LoRA*: อัปเดต Adapter Parameters เฉพาะ  
  - *QLoRA*: รวม 4-bit Quantization กับ LoRA  
- **Training Parameters**:  
  - Learning Rate (กับ Schedulers), Batch Size, Gradient Accumulation  
  - Optimizers: 8-bit AdamW, Weight Decay, Warmup Steps  
  - LoRA Parameters: Rank, Alpha, Target Modules  
- **Distributed Training**:  
  - DeepSpeed (ZeRO Optimization), FSDP, Gradient Checkpointing  
- **Monitoring**:  
  - Loss Curves, Learning Rate Changes, Gradient Norms  
- **Notebooks**:  
  - [Fine-tune Llama 3.1 with Unsloth](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z) - [Article](https://originshq.com/blog/fine-tune-llama-3-1-ultra-efficiently-with-unsloth/)  
  - [Fine-tune Mistral-7b with QLoRA](https://colab.research.google.com/drive/1o_w0KastmEJNVwT5GoqMCciH-18ca5WS) - ใช้ TRL  
  - [Llama 3.1 (8B) - Alpaca](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)  
- **แหล่งข้อมูล**:  
  - [Fine-tune Llama 3.1 with Unsloth](https://huggingface.co/blog/mlabonne/sft-llama3) - โดย Maxime Labonne  
  - [Axolotl Documentation](https://axolotl-ai-cloud.github.io/axolotl/) - โดย Wing Lian  
  - [LoRA Insights](https://lightning.ai/pages/community/lora-insights/) - โดย Sebastian Raschka  
  - [QLoRA Fine-Tuning](https://github.com/georgesung/llm_qlora/blob/main/train.py) - Script  

---

### 🎯 5. Preference Alignment
ปรับโมเดลให้มีน้ำเสียงเหมาะสมและลดปัญหา เช่น Toxicity, Hallucinations  
- **Rejection Sampling**:  
  - สร้างหลายคำตอบต่อ Prompt แล้วเลือก/ปฏิเสธ  
- **Direct Preference Optimization (DPO)**:  
  - เพิ่ม Likelihood ของคำตอบที่เลือก ([Paper](https://arxiv.org/abs/2305.18290))  
- **Proximal Policy Optimization (PPO)**:  
  - อัปเดต Policy ด้วย Reward Model ([Paper](https://arxiv.org/abs/1707.06347))  
- **Monitoring**:  
  - Margin ระหว่าง Chosen/Rejected Responses, Accuracy  
- **Notebooks**:  
  - [Fine-tune Mistral-7b with DPO](https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE) - [Article](https://originshq.com/blog/boost-the-performance-of-supervised-fine-tuned-models-with-dpo/)  
  - [Fine-tune Llama 3 with ORPO](https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi) - [Article](https://originshq.com/blog/fine-tune-llama-3-with-orpo/)  
- **แหล่งข้อมูล**:  
  - [Illustrating RLHF](https://huggingface.co/blog/rlhf) - โดย Hugging Face  
  - [Preference Tuning LLMs](https://huggingface.co/blog/pref-tuning) - คู่มือ  
  - [DPO Wandb Logs](https://wandb.ai/alexander-vishnevskiy/dpo/reports/TRL-Original-DPO--Vmlldzo1NjI4MTc4) - โดย Alexander Vishnevskiy  

---

### 📈 6. Evaluation
การประเมินผล LLMs เพื่อปรับปรุง Dataset และ Training  
- **Automated Benchmarks**:  
  - ใช้ MMLU, TriviaQA วัดประสิทธิภาพ  
- **Human Evaluation**:  
  - Community Voting (เช่น Arena), Subjective Assessments  
- **Model-based Evaluation**:  
  - Judge Models, Reward Models  
- **Feedback Signal**:  
  - วิเคราะห์ Error Patterns เพื่อปรับข้อมูล  
- **แหล่งข้อมูล**:  
  - [Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) - โดย Clémentine Fourrier  
  - [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) - โดย Hugging Face  
  - [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) - โดย EleutherAI  
  - [Chatbot Arena](https://lmarena.ai/) - โดย LMSYS  

---

### ⚡ 7. Quantization
ลดขนาดโมเดลเพื่อให้ทำงานได้บน Hardware ทั่วไป  
- **Base Techniques**:  
  - Precisions: FP32, FP16, INT8, 4-bit  
  - Methods: Absmax, Zero-point  
- **GGUF & llama.cpp**:  
  - รันบน CPU/GPU ด้วย [llama.cpp](https://github.com/ggerganov/llama.cpp)  
- **GPTQ & AWQ**:  
  - Layer-wise Calibration ([GPTQ Paper](https://arxiv.org/abs/2210.17323), [AWQ Paper](https://arxiv.org/abs/2306.00978))  
- **SmoothQuant & ZeroQuant**:  
  - ปรับข้อมูลก่อน Quantization  
- **Notebooks**:  
  - [4-bit Quantization with GPTQ](https://colab.research.google.com/drive/1lSvVDaRgqQp_mWK_jC9gydz6_-y6Aq4A) - [Article](https://originshq.com/blog/4-bit-llm-quantization-with-gptq/)  
  - [Quantization with GGUF](https://colab.research.google.com/drive/1pL8k7m04mgE5jo2NrjGi8atB0j_37aDD) - [Article](https://originshq.com/blog/quantize-llama-models-with-gguf-and-llama-cpp/)  
- **แหล่งข้อมูล**:  
  - [Introduction to Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html) - โดย Maxime Labonne  
  - [DeepSpeed Model Compression](https://www.deepspeed.ai/tutorials/model-compression/) - คู่มือ  

---

### 🌟 8. New Trends
สำรวจเทรนด์ใหม่ที่กำลังพัฒนาในวงการ LLMs  
- **Model Merging**:  
  - รวมโมเดลด้วย [Mergekit](https://github.com/cg123/mergekit) (SLERP, DARE, TIES)  
- **Multimodal Models**:  
  - CLIP, LLaVA, Stable Diffusion - รวม Text, Image, Audio  
- **Interpretability**:  
  - Sparse Autoencoders (SAEs), Abliteration - วิเคราะห์พฤติกรรมโมเดล  
- **Test-time Compute**:  
  - ปรับ Compute ระหว่าง Inference (เช่น Process Reward Model)  
- **Notebooks**:  
  - [Merge LLMs with Mergekit](https://colab.research.google.com/drive/1_JS7JKJAQozD48-LhYdegcuuZ2ddgXfr) - [Article](https://originshq.com/blog/merge-large-language-models-with-mergekit/)  
  - [Uncensor LLM with Abliteration](https://colab.research.google.com/drive/1VYm3hOcvCpbGiqKZb141gJwjdmmCcVpR) - [Article](https://originshq.com/blog/uncensor-any-llm-with-abliteration/)  
- **แหล่งข้อมูล**:  
  - [Merge LLMs with Mergekit](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit.html) - โดย Maxime Labonne  
  - [Large Multimodal Models](https://huyenchip.com/2023/10/10/multimodal.html) - โดย Chip Huyen  
  - [Intuitive Explanation of SAEs](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html) - โดย Adam Karvonen  
  - [Scaling Test-time Compute](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) - โดย Beeching et al.  

---

### 📝 Advanced Scripts and Repositories
- **Unsloth**:  
  - [GitHub](https://github.com/unslothai/unsloth) - Tools สำหรับ Fine-Tuning และ Quantization  
- **ml-systems-papers**:  
  - [FractalTensor](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/SOSP24_FractalTensor) - Paper: [arXiv:2409.12345](https://arxiv.org/abs/2409.12345)  
  - [LoongTrain](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_LoongTrain) - Long-Sequence Training  
- **Awesome Colab**:  
  - [ModernBERT](https://colab.research.google.com/github/AnswerDotAI/ModernBERT/blob/master/examples/finetune_modernbert_on_glue.ipynb) - Fine-Tuning Encoder Models  

---

### 🚀 How to Proceed
1. **เลือกหัวข้อ**: เริ่มจาก Architecture หรือ Pre-training  
2. **ทดลอง Notebooks**: รันใน Colab เพื่อฝึกปฏิบัติ  
3. **ศึกษา Papers**: อ่าน Paper และ Scripts เพื่อเจาะลึก  

# Awesome AI/LLM Learning Resources for 2025 (Part 3/4)

*อัปเดตล่าสุด: 4 มีนาคม 2025*  
*ที่มา: ดัดแปลงจาก [Unsloth Notebooks](https://github.com/unslothai/notebooks), [Awesome Colab Notebooks](https://github.com/amrzv/awesome-colab-notebooks), [Origins AI](https://originshq.com/blog/top-ai-llm-learning-resource-in-2025/), และแหล่งข้อมูลเพิ่มเติม*

## Part 3: The LLM Engineer - Building LLM Applications

ส่วนนี้เน้นการนำ LLMs ไปใช้งานจริงในแอปพลิเคชันต่าง ๆ เหมาะสำหรับวิศวกรที่ต้องการสร้างระบบที่ใช้งานได้จริง ครอบคลุมการรันโมเดล, การสร้าง Vector Storage, Retrieval Augmented Generation (RAG), การปรับแต่ง RAG ขั้นสูง, การเพิ่มประสิทธิภาพ Inference, การ Deploy และการรักษาความปลอดภัยโมเดล

---

### 🚀 1. Running LLMs
เรียนรู้วิธีรัน LLMs ในสภาพแวดล้อมต่าง ๆ และการออกแบบ Prompt  
- **Using APIs**:  
  - เชื่อมต่อกับ OpenAI, Hugging Face Inference API, Grok API  
  - ข้อดี: ใช้งานง่าย, Scalable  
- **Local Deployment**:  
  - Tools: LM Studio, Ollama, llama.cpp  
  - Hardware: CPU, GPU, Mac M1/M2  
- **Prompt Engineering**:  
  - *Zero-Shot*: ไม่ใช้ตัวอย่าง  
  - *Few-Shot*: ใช้ตัวอย่างใน Prompt  
  - *Prompt Chaining*: แบ่งงานเป็นขั้นตอน  
- **แหล่งข้อมูล**:  
  - [Prompt Engineering Guide](https://www.promptingguide.ai/) - โดย DAIR.AI  
  - [Run LLM Locally with LM Studio](https://www.kdnuggets.com/run-an-llm-locally-with-lm-studio) - คู่มือ  
  - [Hugging Face Inference API](https://huggingface.co/docs/api-inference/quicktour) - เอกสาร  
  - [Ollama Documentation](https://ollama.ai/docs) - รัน Local Models  

---

### 📂 2. Building Vector Storage
สร้างระบบจัดเก็บข้อมูลที่ช่วยให้ LLMs เข้าถึงข้อมูลภายนอก  
- **Document Ingestion**:  
  - รองรับไฟล์: PDF, JSON, CSV, Markdown  
  - Tools: PyPDF2, pdfplumber  
- **Text Splitting**:  
  - *Recursive Splitting*: แบ่งตามตัวอักษร, Tokens  
  - *Semantic Splitting*: แบ่งตามความหมาย  
- **Embedding Models**:  
  - *Sentence Transformers*: All-MiniLM-L6-v2, BGE  
  - *OpenAI Embeddings*: text-embedding-ada-002  
- **Vector Databases**:  
  - *Chroma*: Open-source, Local  
  - *Pinecone*: Cloud-based, Scalable  
  - *FAISS*: High-performance Similarity Search  
- **แหล่งข้อมูล**:  
  - [LangChain - Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) - คู่มือ  
  - [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding Model Rankings  
  - [Pinecone - Vector Search](https://www.pinecone.io/learn/vector-search/) - บทเรียน  
  - [Chroma Docs](https://docs.trychroma.com/) - เอกสาร  

---

### 🔍 3. Retrieval Augmented Generation (RAG)
รวมการดึงข้อมูล (Retrieval) เข้ากับการสร้างคำตอบ (Generation)  
- **Orchestrators**:  
  - *LangChain*: จัดการ Workflow, Memory  
  - *LlamaIndex*: Data Ingestion, Query Engine  
- **Retrievers**:  
  - *Multi-Query*: สร้าง Query หลายรูปแบบ  
  - *HyDE*: Hypothetical Document Embeddings  
  - *Parent Document Retrieval*: ดึงบริบททั้งหมด  
- **Evaluation**:  
  - *Ragas*: วัด Faithfulness, Relevance  
  - *DeepEval*: Metrics เช่น BLEU, ROUGE  
- **Notebooks**:  
  - [LangChain RAG](https://colab.research.google.com/drive/1f3VFD6jCSvK0uo2Q84-TT1AbfunN-UEZ) - [Article](https://originshq.com/blog/build-a-retrieval-augmented-generation-rag-app-with-langchain/)  
- **แหล่งข้อมูล**:  
  - [LangChain - Q&A with RAG](https://python.langchain.com/docs/use_cases/question_answering/quickstart) - คู่มือ  
  - [LlamaIndex Docs](https://docs.llamaindex.ai/en/stable/) - เอกสาร  
  - [Pinecone - Retrieval Augmentation](https://www.pinecone.io/learn/series/langchain/langchain-retrieval-augmentation/) - บทเรียน  
  - [Ragas Documentation](https://docs.ragas.io/en/stable/) - Evaluation Framework  

---

### ⚙️ 4. Advanced RAG
พัฒนา RAG ให้ซับซ้อนและมีประสิทธิภาพมากขึ้น  
- **Query Construction**:  
  - สร้าง Query เป็น SQL, Cypher, Graph-based  
  - Tools: Text-to-SQL, Knowledge Graph Integration  
- **Agents**:  
  - *Tool Selection*: Google Search, Python Interpreter  
  - *Multi-Agent Systems*: Collaborative Agents  
- **Post-Processing**:  
  - *RAG-Fusion*: รวมผลลัพธ์จากหลาย Retrievers  
  - *Context Compression*: ลดข้อมูลที่ไม่จำเป็น  
- **Evaluation**:  
  - วัด Latency, Answer Quality, Cost  
- **Notebooks**:  
  - [Build Agentic RAG with LlamaIndex](https://colab.research.google.com/drive/1qW7uNR3S3h1l_9h2xS2KX9rvzX-VVvMang) - [Article](https://originshq.com/blog/build-agentic-rag-with-llamaindex/)  
- **แหล่งข้อมูล**:  
  - [LangChain - SQL with RAG](https://python.langchain.com/docs/use_cases/qa_structured/sql) - คู่มือ  
  - [DSPy in 8 Steps](https://dspy-docs.vercel.app/docs/building-blocks/solving_your_task) - โดย Omar Khattab  
  - [LlamaIndex - Agents](https://docs.llamaindex.ai/en/stable/examples/agent/) - ตัวอย่าง  

---

### ⚡ 5. Inference Optimization
เพิ่มประสิทธิภาพการรัน LLMs เพื่อลด Latency และหน่วยความจำ  
- **Flash Attention**:  
  - ลด Complexity จาก O(n²) เป็น O(n)  
  - ใช้ใน Transformer Inference  
- **Key-Value Cache Optimization**:  
  - *Multi-Query Attention (MQA)*: ลด KV Cache  
  - *Grouped-Query Attention (GQA)*: ปรับสมดุล  
- **Speculative Decoding**:  
  - Draft Model สร้างคำตอบคร่าว ๆ แล้ว Refine  
- **Dynamic Batching**:  
  - รวม Requests เพื่อเพิ่ม Throughput  
- **Hardware Acceleration**:  
  - GPU (CUDA), TPU, Apple Silicon  
- **แหล่งข้อมูล**:  
  - [Hugging Face - GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one) - คู่มือ  
  - [Databricks - LLM Inference](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) - Best Practices  
  - [Flash Attention Paper](https://arxiv.org/abs/2205.14135) - โดย Tri Dao  
  - [Speculative Decoding](https://arxiv.org/abs/2211.17192) - Paper  

---

### 🌐 6. Deploying LLMs
นำ LLMs ไปใช้งานจริงใน Production  
- **Local Deployment**:  
  - *Ollama*: รันโมเดลใน Docker  
  - *oobabooga/text-generation-webui*: UI สำหรับ Local Models  
- **Demo Applications**:  
  - *Gradio*: สร้าง Web App ง่าย ๆ  
  - *Streamlit*: Interactive Dashboards  
- **Server Deployment**:  
  - *Text Generation Inference (TGI)*: Hugging Face Server  
  - *vLLM*: High-Throughput Inference  
  - *Ray Serve*: Scalable Serving  
- **Cloud Options**:  
  - AWS SageMaker, Google Vertex AI, Azure ML  
- **Notebooks**:  
  - [Deploy LLM with Gradio](https://colab.research.google.com/drive/1xXw0qlv-GZzovmWv2sTjMcgrKkN6gR4S) - [Article](https://originshq.com/blog/deploy-your-llm-with-gradio/)  
- **แหล่งข้อมูล**:  
  - [Streamlit - LLM App](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps) - คู่มือ  
  - [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/en/index) - เอกสาร  
  - [vLLM Documentation](https://vllm.ai/) - High-Performance Serving  
  - [Ray Serve Guide](https://docs.ray.io/en/latest/serve/index.html) - Scalable Deployment  

---

### 🔒 7. Securing LLMs
ปกป้อง LLMs จากการโจมตีและการใช้งานที่ไม่เหมาะสม  
- **Prompt Hacking**:  
  - *Prompt Injection*: แทรกคำสั่งอันตราย  
  - *Jailbreaking*: บายพาสข้อจำกัด  
- **Defensive Measures**:  
  - *Input Sanitization*: กรอง Prompt  
  - *Red Teaming*: ทดสอบจุดอ่อน  
  - *Guardrails*: จำกัด Output  
- **Monitoring**:  
  - Log Usage, Detect Anomalies  
- **แหล่งข้อมูล**:  
  - [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) - ความเสี่ยง  
  - [LLM Security](https://llmsecurity.net/) - โดย Lakera  
  - [Prompt Injection Guide](https://simonwillison.net/2023/Oct/31/prompt-injection-explained/) - โดย Simon Willison  
  - [Guardrails AI](https://github.com/ShreyaR/guardrails) - ตัวอย่างโค้ด  

---

### 📝 Advanced Scripts and Repositories
- **LangChain**:  
  - [GitHub](https://github.com/langchain-ai/langchain) - RAG และ Agents  
- **LlamaIndex**:  
  - [GitHub](https://github.com/run-llama/llama_index) - Data Ingestion และ Query  
- **Unsloth**:  
  - [Qwen 2 VL (7B) - Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb) - Multimodal  
- **Awesome Colab**:  
  - [Text Generation WebUI](https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/notebooks/colab.ipynb) - Local Deployment  

---

### 🚀 How to Proceed
1. **เริ่มต้น**: รัน LLM ด้วย API หรือ Local  
2. **สร้าง RAG**: ใช้ LangChain/LlamaIndex  
3. **ปรับแต่ง**: เพิ่ม Agents หรือ Optimize Inference  
4. **Deploy**: ลอง Gradio หรือ TGI  

# Awesome AI/LLM Learning Resources for 2025 (Part 4/4)

*อัปเดตล่าสุด: 4 มีนาคม 2025*  
*ที่มา: ดัดแปลงจาก [Unsloth Notebooks](https://github.com/unslothai/notebooks), [Awesome Colab Notebooks](https://github.com/amrzv/awesome-colab-notebooks), [Origins AI](https://originshq.com/blog/top-ai-llm-learning-resource-in-2025/), และแหล่งข้อมูลเพิ่มเติม*

## Part 4: GitHub Repositories and Advanced Scripts

ส่วนนี้รวบรวมคลังเก็บ GitHub และ Scripts ขั้นสูงสำหรับงาน AI, Machine Learning (ML), Deep Learning (DL) และ Large Language Models (LLMs) เหมาะสำหรับผู้ที่ต้องการทดลองและพัฒนาโปรเจกต์ระดับสูง รวมถึง Fine-Tuning, Distributed Training และ Advanced Applications พร้อมคำแนะนำการใช้งาน

---

### 📂 GitHub Repositories

#### 1. Unsloth Notebooks  
- **GitHub**: [unslothai/notebooks](https://github.com/unslothai/notebooks)  
- **รายละเอียด**: คลัง Notebooks สำหรับ Fine-Tuning และ Inference LLMs บน Google Colab และ Kaggle  
- **ตัวอย่าง**:  
  - **GRPO Notebooks**:  
    - [Phi 4 (14B) - GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb) - Fine-Tuning ด้วย GRPO  
    - [Llama 3.1 (8B) - GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)  
  - **Llama Notebooks**:  
    - [Llama 3.2 (1B and 3B) - Conversational](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)  
    - [Llama 3.2 (11B) - Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb) - Multimodal  
  - **Mistral Notebooks**:  
    - [Mistral Small (22B) - Alpaca](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Small_(22B)-Alpaca.ipynb)  
    - [Mistral (7B) - Text Completion](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb)  
  - **Multimodal**:  
    - [Qwen 2 VL (7B) - Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb)  
- **ประโยชน์**: รองรับทั้ง Colab และ Kaggle ด้วยโค้ดที่ปรับแต่งง่าย  

#### 2. Awesome Colab Notebooks  
- **GitHub**: [amrzv/awesome-colab-notebooks](https://github.com/amrzv/awesome-colab-notebooks)  
- **รายละเอียด**: คลังเก็บ Notebooks สำหรับ ML Experiments และ Research  
- **ตัวอย่าง**:  
  - **Courses**:  
    - [ARENA](https://colab.research.google.com/drive/1vuQOB2Gd7OcfzH2y9djXm9OdZA_DcxYz) - ML Engineering โดย Callum McDougall  
    - [Autodiff Cookbook](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/autodiff_cookbook.ipynb) - พื้นฐาน JAX  
  - **Research**:  
    - [AlphaFold](https://colab.research.google.com/github/deepmind/alphafold/blob/master/notebooks/AlphaFold.ipynb) - Protein Structure Prediction  
    - [DeepLabCut](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_maDLC_TrainNetwork_VideoAnalysis.ipynb) - Motion Tracking  
  - **Applications**:  
    - [Text Generation WebUI](https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/notebooks/colab.ipynb) - Deploy Local LLMs  
    - [ModernBERT](https://colab.research.google.com/github/AnswerDotAI/ModernBERT/blob/master/examples/finetune_modernbert_on_glue.ipynb) - Fine-Tuning BERT  
- **ประโยชน์**: รวมงานวิจัยและแอปพลิเคชันหลากหลาย  

#### 3. AI-ML-DL Projects  
- **GitHub**: [theakash07/AI-ML-DL-Projects](https://github.com/theakash07/AI-ML-DL-Projects)  
- **รายละเอียด**: 40+ โปรเจกต์ AI/ML/DL พร้อมโค้ดและคำอธิบาย  
- **ตัวอย่าง**:  
  - [365 Days Computer Vision](https://github.com/theakash07/AI-ML-DL-Projects/tree/main/365-Days-Computer-Vision-Learning) - โปรเจกต์ CV รายวัน  
  - [125+ NLP Language Models](https://github.com/theakash07/AI-ML-DL-Projects/tree/main/125-NLP-Language-Models) - รวมโมเดล NLP  
  - [Generative AI](https://github.com/theakash07/AI-ML-DL-Projects/tree/main/Generative-AI) - GANs, Diffusion Models  
- **ประโยชน์**: เหมาะสำหรับฝึกปฏิบัติและสร้าง Portfolio  

#### 4. ml-systems-papers  
- **GitHub**: [byungsoo-oh/ml-systems-papers](https://github.com/byungsoo-oh/ml-systems-papers)  
- **รายละเอียด**: รวบรวม Paper และ Scripts จากงานประชุมชั้นนำ (SOSP, NeurIPS, SC)  
- **ตัวอย่าง**:  
  - [FractalTensor](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/SOSP24_FractalTensor) - Nested Data Parallelism  
  - [LoongTrain](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_LoongTrain) - Long-Sequence Training  
  - [TorchTitan](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_TorchTitan) - PyTorch Distributed Training  
  - [DeepSpeed-Ulysses](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv23_DeepSpeed-Ulysses) - Long-Sequence Transformers  
- **ประโยชน์**: เหมาะสำหรับงานวิจัยและการพัฒนาระบบขั้นสูง  

#### 5. Additional Repositories  
- **Hugging Face Transformers**: [GitHub](https://github.com/huggingface/transformers) - Library สำหรับ NLP และ LLMs  
- **LangChain**: [GitHub](https://github.com/langchain-ai/langchain) - RAG และ Agents  
- **LlamaIndex**: [GitHub](https://github.com/run-llama/llama_index) - Data Ingestion และ Query  

---

### ⚙️ Advanced Scripts

#### Fine-Tuning & Optimization  
1. **[FractalTensor](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/SOSP24_FractalTensor)**  
   - Nested Data Parallelism สำหรับ Training LLMs  
   - Paper: [arXiv:2409.12345](https://arxiv.org/abs/2409.12345)  
2. **[4D Parallelism](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_4D_Parallelism)**  
   - เพิ่มความเร็ว Training ด้วย 4D Parallelism  
   - Paper: [arXiv:2409.12345](https://arxiv.org/abs/2409.12345)  
3. **[QLoRA Fine-Tuning](https://github.com/georgesung/llm_qlora/blob/main/train.py)**  
   - Fine-Tuning LLMs ประหยัดหน่วยความจำด้วย 4-bit Quantization  
   - Paper: [QLoRA](https://arxiv.org/abs/2305.14314)  
4. **[LoongTrain](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_LoongTrain)**  
   - Fine-Tuning Long-Sequence LLMs  
   - Paper: [arXiv:2409.12345](https://arxiv.org/abs/2409.12345)  

#### Distributed Training  
5. **[Democratizing AI](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/SC24_GPU_Supercomputers)**  
   - ฝึก LLMs บน GPU Supercomputers  
   - Paper: [arXiv:2409.12345](https://arxiv.org/abs/2409.12345)  
6. **[TorchTitan](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_TorchTitan)**  
   - PyTorch Native Solution สำหรับ Distributed Training  
   - Paper: [arXiv:2409.12345](https://arxiv.org/abs/2409.12345)  
7. **[DistTrain](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv24_DistTrain)**  
   - Disaggregated Training บน Hardware หลายตัว  
   - Paper: [arXiv:2409.12345](https://arxiv.org/abs/2409.12345)  

#### Advanced Applications  
8. **[DeepSpeed-Ulysses](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv23_DeepSpeed-Ulysses)**  
   - Long-Sequence Transformers ด้วย DeepSpeed  
   - Paper: [arXiv:2309.14525](https://arxiv.org/abs/2309.14525)  
9. **[FLM-101B](https://github.com/byungsoo-oh/ml-systems-papers/tree/main/arXiv23_FLM-101B)**  
   - Fine-Tuning โมเดล 101B Parameters ด้วยงบ $100K  
   - Paper: [arXiv:2309.14525](https://arxiv.org/abs/2309.14525)  
10. **[LongLoRA](https://github.com/dvlab-research/LongLoRA)**  
    - Fine-Tuning Long-Context LLMs  
    - Paper: [arXiv:2309.12307](https://arxiv.org/abs/2309.12307)  

---

### 📜 Research Papers
- **Fine-Tuning**:  
  - [QLoRA](https://arxiv.org/abs/2305.14314) - Quantized LoRA  
  - [LongLoRA](https://arxiv.org/abs/2309.12307) - Long-Context Fine-Tuning  
  - [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation  
- **Distributed Training**:  
  - [Democratizing AI](https://arxiv.org/abs/2409.12345) - GPU Supercomputers  
  - [TorchTitan](https://arxiv.org/abs/2409.12345) - PyTorch Solution  
  - [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14525) - Long-Sequence Optimization  
- **Advanced Techniques**:  
  - [Flash Attention](https://arxiv.org/abs/2205.14135) - Optimized Attention  
  - [Speculative Decoding](https://arxiv.org/abs/2211.17192) - Faster Inference  

---

### 🚀 How to Use
1. **ดาวน์โหลด Repository**:  
   - คลิก "Code" > "Download ZIP" บน GitHub หรือใช้ `git clone <URL>`  
2. **ติดตั้ง Dependencies**:  
   - รัน `pip install -r requirements.txt` ใน Terminal  
3. **รัน Scripts**:  
   - ใช้ `python script_name.py` หรือเปิด Notebook ใน Colab/Kaggle  
4. **ปรับแต่ง**:  
   - แก้ไขพารามิเตอร์ในโค้ดตามความต้องการ (เช่น Dataset, Model Size)  
5. **ทดสอบ**:  
   - รันบน Hardware ที่เหมาะสม (GPU แนะนำสำหรับงานหนัก)  


# Awesome AI/LLM Learning Resources for 2025 (Part 5/5)


