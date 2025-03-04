*อัปเดตล่าสุด: 4 มีนาคม 2025*  
*ที่มา: [DeepSeek AI GitHub Repositories](https://github.com/orgs/deepseek-ai/repositories)*

## Part 5: DeepSeek AI GitHub Repositories

### 📂 DeepSeek AI Repositories

#### 1. DeepEP  
- **URL**: [https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)  
- **คำอธิบาย**: ไลบรารีการสื่อสารแบบ Expert-Parallel ที่มีประสิทธิภาพสูง ช่วยจัดการการสื่อสารระหว่างโมเดลในระบบฝึก AI ขนาดใหญ่  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ใช้กลไก Expert-Parallel เพื่อแบ่งงานฝึกโมเดลให้กระจายไปยัง GPU หลายตัว ลดการติดขัดในการสื่อสารระหว่างอุปกรณ์  
  - **วิธีใช้**: ดาวน์โหลดโค้ด, ติดตั้ง Dependencies (เช่น PyTorch), รวมเข้ากับ Pipeline การฝึกโมเดล โดยกำหนด Expert Modules ใน Config  

#### 2. 3FS  
- **URL**: [https://github.com/deepseek-ai/3FS](https://github.com/deepseek-ai/3FS)  
- **คำอธิบาย**: ระบบไฟล์กระจายประสิทธิภาพสูง ออกแบบมาเพื่อรองรับการฝึกและ Inference AI โดยเฉพาะ  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: จัดเก็บและเข้าถึงข้อมูลแบบกระจาย (Distributed File System) เพื่อลด Latency ในงาน AI ขนาดใหญ่  
  - **วิธีใช้**: ติดตั้งผ่าน Docker หรือ Source Code, กำหนด Cluster Configuration, ใช้คู่กับ Framework เช่น TensorFlow หรือ PyTorch  

#### 3. DeepGEMM  
- **URL**: [https://github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)  
- **คำอธิบาย**: Kernel GEMM แบบ FP8 ที่สะอาดและมีประสิทธิภาพ รองรับการปรับขนาดแบบละเอียด (Fine-grained Scaling)  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ปรับปรุงการคำนวณ Matrix Multiplication ด้วย FP8 Precision เพื่อประหยัดหน่วยความจำและเพิ่มความเร็ว  
  - **วิธีใช้**: รวม Kernel เข้ากับโมเดล Deep Learning, คอมไพล์ด้วย CUDA, เรียกใช้ใน Layer ที่ต้องการ GEMM  

#### 4. open-infra-index  
- **URL**: [https://github.com/deepseek-ai/open-infra-index](https://github.com/deepseek-ai/open-infra-index)  
- **คำอธิบาย**: เครื่องมือโครงสร้างพื้นฐาน AI ที่ผ่านการทดสอบใน Production เพื่อพัฒนา AGI และนวัตกรรมชุมชน  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: รวบรวมเครื่องมือ Open-source ที่ผ่านการทดสอบจริงสำหรับงาน AGI  
  - **วิธีใช้**: เลือกเครื่องมือจาก Index, ดาวน์โหลดตามลิงก์ใน README, ปรับใช้ใน Workflow ของคุณ  

#### 5. profile-data  
- **URL**: [https://github.com/deepseek-ai/profile-data](https://github.com/deepseek-ai/profile-data)  
- **คำอธิบาย**: วิเคราะห์การทับซ้อนระหว่างการคำนวณและการสื่อสารใน DeepSeek-V3/R1  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: สร้างโปรไฟล์การทำงานของ V3/R1 เพื่อหาจุดที่สามารถปรับปรุงประสิทธิภาพ  
  - **วิธีใช้**: รัน Script วิเคราะห์กับ Log การฝึก, ใช้ผลลัพธ์ปรับ Hyperparameters หรือ Pipeline  

#### 6. awesome-deepseek-integration  
- **URL**: [https://github.com/deepseek-ai/awesome-deepseek-integration](https://github.com/deepseek-ai/awesome-deepseek-integration)  
- **คำอธิบาย**: รวมวิธีผสาน DeepSeek API เข้ากับซอฟต์แวร์ยอดนิยม เช่น IDEs และแพลตฟอร์มอื่น ๆ  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: จัดเตรียมโค้ดตัวอย่างสำหรับเชื่อมต่อ DeepSeek API กับแอปพลิเคชัน  
  - **วิธีใช้**: เลือกซอฟต์แวร์เป้าหมาย (เช่น VS Code), คัดลอกโค้ดจากตัวอย่าง, ปรับแต่งด้วย API Key  

#### 7. smallpond  
- **URL**: [https://github.com/deepseek-ai/smallpond](https://github.com/deepseek-ai/smallpond)  
- **คำอธิบาย**: เฟรมเวิร์กประมวลผลข้อมูลน้ำหนักเบา สร้างบน DuckDB และ 3FS  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ใช้ DuckDB สำหรับ Query และ 3FS สำหรับจัดเก็บข้อมูลแบบกระจาย  
  - **วิธีใช้**: ติดตั้ง DuckDB และ 3FS, รัน Script ตัวอย่างใน README, ป้อนข้อมูลเพื่อประมวลผล  

#### 8. FlashMLA  
- **URL**: [https://github.com/deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)  
- **คำอธิบาย**: Kernel ถอดรหัส MLA (Multi-head Latent Attention) ที่มีประสิทธิภาพสูง  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ลดความซับซ้อนของ Attention Mechanism ด้วย Kernel ที่เร็วขึ้น  
  - **วิธีใช้**: คอมไพล์ Kernel ด้วย CUDA, รวมเข้ากับโมเดล Transformer, ทดสอบ Inference  

#### 9. DualPipe  
- **URL**: [https://github.com/deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe)  
- **คำอธิบาย**: อัลกอริทึม Pipeline Parallelism แบบสองทิศทาง เพื่อการทับซ้อนการคำนวณ-สื่อสารใน V3/R1  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ฝึกโมเดลแบบ Pipeline โดยให้ Forward และ Backward Pass ทับซ้อนกัน  
  - **วิธีใช้**: ปรับ Config การฝึกใน V3/R1, รวมโค้ด DualPipe, รันบน Multi-GPU  

#### 10. EPLB  
- **URL**: [https://github.com/deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)  
- **คำอธิบาย**: Expert Parallelism Load Balancer สำหรับกระจายงานในระบบฝึกโมเดล  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ปรับสมดุลโหลดงานระหว่าง Experts เพื่อลดการรอคอย  
  - **วิธีใช้**: รวม EPLB เข้ากับ Framework การฝึก, กำหนดจำนวน Experts ใน Config  

#### 11. DeepSeek-VL2  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)  
- **คำอธิบาย**: โมเดล Mixture-of-Experts Vision-Language สำหรับการเข้าใจหลายรูปแบบขั้นสูง  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: รวม Vision และ Language Processing ด้วย MoE Architecture  
  - **วิธีใช้**: ดาวน์โหลดโมเดล, รัน Inference ด้วยภาพและข้อความตามตัวอย่างใน README  

#### 12. DeepSeek-V3  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)  
- **คำอธิบาย**: โมเดล Mixture-of-Experts ขนาด 671B Parameters (37B ใช้งานต่อ Token) ที่มีประสิทธิภาพสูง  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ใช้ MoE เพื่อเลือก Experts ที่เหมาะสมต่อการตอบคำถาม  
  - **วิธีใช้**: ดาวน์โหลด Weights, รัน Inference หรือ Fine-Tune ด้วย Script ที่ให้มา  

#### 13. DeepSeek-R1  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)  
- **คำอธิบาย**: โมเดล Reasoning รุ่นแรกที่ฝึกด้วย Reinforcement Learning มีความสามารถเทียบเท่า OpenAI-o1  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ฝึกด้วย RL เพื่อแก้ปัญหาเชิง Reasoning เช่น คณิตศาสตร์และโค้ด  
  - **วิธีใช้**: รันโมเดลด้วย Prompt ที่ซับซ้อน, ใช้ API หรือ Local Inference  

#### 14. Janus  
- **URL**: [https://github.com/deepseek-ai/Janus](https://github.com/deepseek-ai/Janus)  
- **คำอธิบาย**: โมเดล Multimodal รุ่น Janus-Series สำหรับการเข้าใจและสร้างข้อมูลหลายรูปแบบ  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: รวม Text, Image และข้อมูลอื่น ๆ ในโมเดลเดียว  
  - **วิธีใช้**: ดาวน์โหลด Janus-Series, ทดสอบด้วย Input หลายรูปแบบตามตัวอย่าง  

#### 15. DeepSeek-V2  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)  
- **คำอธิบาย**: โมเดล Mixture-of-Experts ที่แข็งแกร่ง, ประหยัด, และมีประสิทธิภาพ (236B Parameters)  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ลดต้นทุนการคำนวณด้วย MoE ในงานภาษาทั่วไป  
  - **วิธีใช้**: ใช้ Weights ที่ให้มา, รัน Inference หรือ Fine-Tune บน Dataset เฉพาะ  

#### 16. DeepSeek-Coder-V2  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2)  
- **คำอธิบาย**: โมเดลรหัสที่ทลายกำแพง Closed-source ในด้าน Code Intelligence  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ฝึกเพื่อเข้าใจและสร้างโค้ดในหลายภาษาโปรแกรม  
  - **วิธีใช้**: รันโมเดลด้วย Prompt โค้ด, ใช้ใน IDE หรือผ่าน API  

#### 17. ESFT  
- **URL**: [https://github.com/deepseek-ai/ESFT](https://github.com/deepseek-ai/ESFT)  
- **คำอธิบาย**: Expert Specialized Fine-Tuning เทคนิคการปรับแต่งโมเดลสำหรับผู้เชี่ยวชาญ  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ปรับแต่ง Experts ใน MoE ให้เชี่ยวชาญงานเฉพาะ  
  - **วิธีใช้**: รัน Script Fine-Tuning ด้วย Dataset เป้าหมายตาม README  

#### 18. DreamCraft3D  
- **URL**: [https://github.com/deepseek-ai/DreamCraft3D](https://github.com/deepseek-ai/DreamCraft3D)  
- **คำอธิบาย**: การใช้งาน DreamCraft3D (ICLR 2024) สำหรับสร้าง 3D แบบลำดับขั้นด้วย Diffusion Prior  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ใช้ Diffusion Model สร้างโมเดล 3D จาก Text หรือ Image  
  - **วิธีใช้**: ติดตั้ง Dependencies, รันโค้ดด้วย Input เช่น Text Prompt  

#### 19. DeepSeek-Prover-V1.5  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5)  
- **คำอธิบาย**: โมเดลสำหรับพิสูจน์คณิตศาสตร์  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ฝึกเพื่อพิสูจน์ทฤษฎีคณิตศาสตร์และ Reasoning  
  - **วิธีใช้**: รันโมเดลด้วยโจทย์คณิตศาสตร์, ตรวจสอบ Output ตามตัวอย่าง  

#### 20. DeepSeek-Coder  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder)  
- **คำอธิบาย**: DeepSeek Coder: "Let the Code Write Itself" โมเดลเขียนโค้ดอัตโนมัติ  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: สร้างโค้ดจาก Prompt หรือคอมเมนต์ในหลายภาษา  
  - **วิธีใช้**: ใช้ API หรือ Local Weights, ป้อน Prompt เพื่อสร้างโค้ด  

#### 21. DeepSeek-VL  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL)  
- **คำอธิบาย**: DeepSeek-VL: โมเดล Vision-Language สำหรับการเข้าใจโลกจริง  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ประมวลผลภาพและข้อความพร้อมกันเพื่อตอบคำถาม  
  - **วิธีใช้**: รัน Inference ด้วยภาพและคำถามตาม Script ตัวอย่าง  

#### 22. DeepSeek-Math  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)  
- **คำอธิบาย**: DeepSeekMath: โมเดลที่ขยายขีดจำกัดการแก้ปัญหาคณิตศาสตร์ใน LLMs  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ฝึกเพื่อแก้โจทย์คณิตศาสตร์และแสดงขั้นตอนการคิด  
  - **วิธีใช้**: รันโมเดลด้วยโจทย์คณิตศาสตร์, ปรับแต่ง Output Format  

#### 23. awesome-deepseek-coder  
- **URL**: [https://github.com/deepseek-ai/awesome-deepseek-coder](https://github.com/deepseek-ai/awesome-deepseek-coder)  
- **คำอธิบาย**: รายการโปรเจกต์ Open-source ที่เกี่ยวข้องกับ DeepSeek Coder  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: รวบรวมทรัพยากรและตัวอย่างการใช้ DeepSeek Coder  
  - **วิธีใช้**: เลือกโปรเจกต์จาก List, ดาวน์โหลดโค้ด, ทดลองใช้งาน  

#### 24. DeepSeek-LLM  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM)  
- **คำอธิบาย**: DeepSeek LLM: "Let there be answers" โมเดลภาษาสำหรับงานทั่วไป  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: โมเดลพื้นฐานสำหรับงาน Text Generation และ Q&A  
  - **วิธีใช้**: รัน Inference ด้วย Prompt หรือ Fine-Tune ตามความต้องการ  

#### 25. DeepSeek-MoE  
- **URL**: [https://github.com/deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE)  
- **คำอธิบาย**: DeepSeekMoE: โมเดล Mixture-of-Experts ที่มุ่งสู่ความเชี่ยวชาญสูงสุด  
- **หลักการทำงานและวิธีใช้**:  
  - **หลักการ**: ใช้ MoE เพื่อแบ่งงานให้ Experts เฉพาะด้าน  
  - **วิธีใช้**: ดาวน์โหลด Weights, รัน Inference หรือฝึกต่อด้วย Script  

---

### 🚀 How to Proceed
1. **เลือกคลัง**: เริ่มจากคลังที่ตรงกับงานของคุณ (เช่น DeepSeek-Coder สำหรับโค้ด, DeepSeek-V3 สำหรับ LLM)  
2. **ดาวน์โหลด**: ใช้ `git clone <URL>` หรือดาวน์โหลด ZIP จาก GitHub  
3. **ติดตั้ง**: รัน `pip install -r requirements.txt` หรือตามคำแนะนำใน README  
4. **ทดลอง**: ใช้ Script หรือ API ตามตัวอย่าง, ปรับแต่งตามความต้องการ  
5. **Hardware**: เตรียม GPU (แนะนำ 16GB+ VRAM) หรือใช้ Cloud เช่น Colab Pro  
