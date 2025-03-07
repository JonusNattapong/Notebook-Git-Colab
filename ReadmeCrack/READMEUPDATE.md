<div align="center">

![Project Logo](./public/Zom.png)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/Apache-2.0-green.svg)](LICENSE)
[![GitHub Package](https://img.shields.io/badge/GitHub-Package-green.svg)](https://github.com/features/packages)
[![GitHub Stars](https://img.shields.io/github/stars/JonusNattapong/Notebook-Git-Colab.svg?style=social)](https://github.com/JonusNattapong/Notebook-Git-Colab/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/JonusNattapong/Notebook-Git-Colab.svg?style=social)](https://github.com/JonusNattapong/Notebook-Git-Colab/network/members)
[![GitHub Followers](https://img.shields.io/github/followers/JonusNattapong.svg?style=social)](https://github.com/JonusNattapong/followers)

</div>

# Notebook-Git-Colab

## ส่วนที่ 1: พื้นฐานที่จำเป็น

*ส่วนนี้เหมาะสำหรับผู้เริ่มต้นที่ต้องการสร้างรากฐานที่มั่นคงในคณิตศาสตร์, Python, โครงข่ายประสาทเทียม และ NLP ก่อนที่จะศึกษาเทคนิคขั้นสูง*

### 1.1 คณิตศาสตร์สำหรับ Machine Learning

| แหล่งข้อมูล                                    | คำอธิบาย                                                                                                   | หัวข้อย่อย (Subtopics)                                                                    | ระดับ (Level)     | ลิงก์                                                                                                                              |
| :------------------------------------------- | :------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- | :---------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| 3Blue1Brown - Linear Algebra                 | ชุดวิดีโอสอนที่ใช้ภาพเคลื่อนไหวอธิบายพีชคณิตเชิงเส้น เหมาะสำหรับเห็นภาพคอนเซปต์ยากๆ เช่น การแปลงเมทริกซ์หรือค่าเจาะจงที่ใช้ใน PCA | เวกเตอร์, เมทริกซ์, การแปลง (transformations), ค่าเจาะจง (eigenvalues - ใช้ใน PCA), เวกเตอร์เจาะจง (eigenvectors - ใช้ใน SVD) | `[Beginner]`      | [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)                                     |
| Khan Academy - Probability and Statistics    | คอร์สออนไลน์ฟรี สอนความน่าจะเป็นและสถิติพื้นฐานที่จำเป็นสำหรับ ML เช่น การแจกแจงที่ใช้ในโมเดลหรือการทดสอบสมมติฐาน | ความน่าจะเป็น, การแจกแจง (distributions), การทดสอบสมมติฐาน, การอนุมานแบบเบย์ (Bayesian inference)      | `[Beginner/Intermediate]` | [Khan Academy Course](https://www.khanacademy.org/math/statistics-probability)                                                     |
| Mathematics for Machine Learning (MML)       | หนังสือเรียนฟรีจาก Cambridge อธิบายคณิตศาสตร์ที่ใช้ใน ML อย่างละเอียด เช่น การหาค่าเหมาะที่สุดสำหรับ gradient descent | พีชคณิตเชิงเส้น, แคลคูลัส, ความน่าจะเป็น, การหาค่าเหมาะที่สุด (optimization)                        | `[Intermediate/Advanced]` | [MML Book](https://mml-book.github.io/)                                                                                       |
| All of Statistics (Wasserman)                | หนังสืออ้างอิงสถิติฉบับสมบูรณ์ เหมาะสำหรับผู้ที่ต้องการเจาะลึก เช่น การวิเคราะห์อนุกรมเวลาใน ML หรือการถดถอยขั้นสูง | การอนุมานทางสถิติ (statistical inference), การทดสอบสมมติฐาน, การถดถอย (regression), อนุกรมเวลา (time series) | `[Advanced]`      | [All of Statistics Book](https://www.stat.cmu.edu/~larry/all-of-statistics/) หรือ [Springer Link](https://link.springer.com/book/10.1007/978-0-387-21736-9) |

### 1.2 การเขียนโปรแกรม Python

| แหล่งข้อมูล                                  | คำอธิบาย                                                                                          | หัวข้อย่อย (Subtopics)                                                                       | ระดับ (Level)        | ลิงก์                                                                                                |
| :------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- | :------------------- | :-------------------------------------------------------------------------------------------------- |
| Python Official Documentation              | เอกสารอ้างอิงอย่างเป็นทางการของ Python อ่านเพื่อเข้าใจพื้นฐานและโมดูลที่ใช้บ่อยใน ML เช่น `random` หรือ `math` | ไวยากรณ์ (syntax), โครงสร้างข้อมูล, โมดูล (modules), ไลบรารีมาตรฐาน (standard library)               | `[All Levels]`   | [Python Docs](https://docs.python.org/3/)                                                               |
| Google's Python Class                      | คอร์สฟรีจาก Google สอน Python พื้นฐาน เหมาะสำหรับผู้เริ่มต้นที่อยากฝึกเขียนโค้ดจริง เช่น การจัดการไฟล์ | พื้นฐาน, สตริง (strings), ลิสต์ (lists), ดิกชันนารี (dictionaries), ไฟล์, regular expressions      | `[Beginner]`       | [Google's Python Class](https://developers.google.com/edu/python/)                                      |
| Corey Schafer - Python Tutorials           | ช่อง YouTube สอน Python เชิงลึก เหมาะสำหรับการเรียนรู้ OOP และงาน data science เช่น การใช้ pandas | พื้นฐาน, การเขียนโปรแกรมเชิงวัตถุ (OOP), การพัฒนาเว็บ, วิทยาศาสตร์ข้อมูล (data science)            | `[Beginner/Intermediate]` | [Corey Schafer YouTube](https://www.youtube.com/c/Coreymschafer)                                        |
| Sentdex - Python Programming Tutorials     | ช่อง YouTube สอน Python แบบประยุกต์ เช่น การสร้างโมเดล ML หรือวิเคราะห์ข้อมูลด้วย NumPy          | Machine Learning, การวิเคราะห์ข้อมูล, การพัฒนาเว็บ, การพัฒนาเกม                                 | `[Intermediate]`    | [Sentdex YouTube](https://www.youtube.com/c/sentdex)                                                 |
| Python for Data Analysis (Wes McKinney)    | หนังสือสอนการใช้ Python (เน้น pandas) สำหรับการวิเคราะห์ข้อมูล เหมาะสำหรับงาน ML และ data science   | pandas, NumPy, การจัดการข้อมูล, การทำความสะอาดข้อมูล, การแสดงภาพข้อมูล (data visualization)     | `[Intermediate/Advanced]` | [Python for Data Analysis](https://wesmckinney.com/book/)                                               |
| Automate the Boring Stuff with Python      | หนังสือและคอร์สฟรี สอน Python พื้นฐานผ่านโปรเจกต์จริง เช่น อัตโนมัติงานเอกสาร เหมาะสำหรับผู้เริ่มต้น   | พื้นฐาน, การจัดการไฟล์, การ scraping เว็บ, การทำงานกับ Excel                                 | `[Beginner]`        | [Automate the Boring Stuff](https://automatetheboringstuff.com/)                                       |

### 1.3 โครงข่ายประสาทเทียม (Neural Networks)

| แหล่งข้อมูล                                    | คำอธิบาย                                                                                                | หัวข้อย่อย (Subtopics)                                                                                   | ระดับ (Level)          | ลิงก์                                                                                                                   |
| :------------------------------------------- | :--------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- | :--------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| DeepLearning.AI - Deep Learning Specialization | ชุดคอร์สออนไลน์บน Coursera อธิบายพื้นฐาน Deep Learning เช่น backpropagation ที่ใช้ในโครงข่ายประสาทเทียม     | โครงข่ายประสาทเทียม, backpropagation, convolutional neural networks (CNNs), recurrent neural networks (RNNs) | `[Beginner/Intermediate]` | [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)                                  |
| 3Blue1Brown - Neural Networks                | ชุดวิดีโอใช้ภาพเคลื่อนไหวอธิบายการทำงานของโครงข่ายประสาทเทียม เช่น gradient descent และเพอร์เซปตรอน       | เพอร์เซปตรอน (perceptrons), ฟังก์ชันกระตุ้น (activation functions), backpropagation, gradient descent   | `[Beginner]`          | [Neural Networks Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)                  |
| fast.ai - Practical Deep Learning for Coders | คอร์สเน้นปฏิบัติ สอน Deep Learning ด้วย fastai (บน PyTorch) เหมาะสำหรับการเริ่มต้นสร้างโมเดลจริง          | การประยุกต์ใช้ Deep Learning, PyTorch, ไลบรารี fastai                                                  | `[Intermediate]`       | [fast.ai Course](https://course.fast.ai/)                                                                            |
| Stanford CS231n - Convolutional Neural Networks | คอร์สจาก Stanford เน้น CNNs สำหรับงาน Computer Vision เช่น การจำแนกภาพ อธิบายสถาปัตยกรรม CNN อย่างละเอียด   | Convolutional layers, pooling layers, สถาปัตยกรรม CNN ยอดนิยม                                       | `[Intermediate/Advanced]` | [CS231n Course](http://cs231n.stanford.edu/)                                                                         |
| Transformers Explained (Hugging Face)        | บทความสั้นจาก Hugging Face อธิบายพื้นฐาน Transformers ซึ่งเป็นรากฐานของ LLM เช่น attention mechanism     | Attention mechanism, self-attention, Transformer architecture                                  | `[Beginner/Intermediate]` | [Transformers Explained](https://huggingface.co/docs/transformers/quicktour)                                         |

### 1.4 การประมวลผลภาษาธรรมชาติ (NLP)

| แหล่งข้อมูล                                                                       | คำอธิบาย                                                                                                  | หัวข้อย่อย (Subtopics)                                                                                                | ระดับ (Level)          | ลิงก์                                                                                                                   |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- | :--------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| Stanford CS224N - NLP with Deep Learning                                       | คอร์สจาก Stanford ครอบคลุม NLP สมัยใหม่ด้วย Deep Learning เช่น การใช้ transformers ในงานแปลภาษา            | Word embeddings, recurrent neural networks (RNNs), transformers, question answering, machine translation  | `[Intermediate/Advanced]` | [CS224N Course](http://web.stanford.edu/class/cs224n/)                                                                   |
| Jurafsky & Martin - Speech and Language Processing (3rd ed. draft)              | หนังสือเรียน NLP ฉบับสมบูรณ์ (ฉบับร่างฟรี) อธิบายตั้งแต่พื้นฐานถึงขั้นสูง เช่น language modeling          | Language modeling, parsing, semantics, discourse, machine translation                                        | `[Intermediate/Advanced]` | [SLP Book (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)                                                       |
| Hugging Face - NLP Course                                                      | คอร์ส interactive สอนการใช้ Transformers library สำหรับงาน NLP เช่น fine-tuning โมเดลสำหรับการจำแนกข้อความ | Transformers, tokenization, fine-tuning, งาน NLP ทั่วไป                                               | `[Beginner/Intermediate]` | [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter1/1)                                                   |
| Natural Language Processing with Python (NLTK Book)                            | หนังสือฟรีสอน NLP ด้วย Python และ NLTK เหมาะสำหรับผู้เริ่มต้นที่อยากฝึกพื้นฐาน เช่น การตัดคำ (tokenization) | Tokenization, part-of-speech tagging, text classification, sentiment analysis                        | `[Beginner]`             | [NLTK Book](https://www.nltk.org/book/)                                                                                 |

### 1.5 แนวทางการเรียนรู้ (How to Proceed)

1. **เริ่มต้นจากพื้นฐาน:** หากคุณเป็นมือใหม่ในวงการ Machine Learning ให้เริ่มด้วยชุดวิดีโอ Linear Algebra ของ 3Blue1Brown และคอร์ส Probability and Statistics ของ Khan Academy เพื่อเข้าใจคณิตศาสตร์พื้นฐาน `[Beginner]`
2. **เรียนรู้ภาษา Python:** ศึกษา Google's Python Class หรือ Automate the Boring Stuff ควบคู่ไปกับการฝึกเขียนโค้ด เช่น สร้างสคริปต์จัดการไฟล์ง่ายๆ `[Beginner/Intermediate]`
3. **ทำความเข้าใจโครงข่ายประสาทเทียม:** เรียน DeepLearning.AI Specialization หรือ fast.ai Course และดูวิดีโอของ 3Blue1Brown เพื่อเห็นภาพการทำงาน ลองสร้างโมเดลจำแนกภาพด้วย PyTorch `[Beginner/Intermediate]`
4. **เจาะลึก NLP:** ศึกษา CS224N ของ Stanford และลองทำ Hugging Face NLP Course เพื่อฝึก fine-tuning โมเดลสำหรับงานจำแนกข้อความ `[Intermediate]`
5. **อ่านหนังสือเพิ่มเติม:** ใช้หนังสือ MML และ Jurafsky & Martin เป็นแหล่งอ้างอิงสำหรับข้อมูลเชิงลึก เช่น การคำนวณ gradient หรือการสร้าง language model `[Intermediate/Advanced]`

**ข้อควรจำ:**
- การเรียนรู้ Machine Learning ต้องใช้เวลาและความสม่ำเสมอ ฝึกทำโปรเจกต์เล็กๆ เช่น จำแนกข้อความหรือวิเคราะห์ข้อมูล เพื่อทดสอบความเข้าใจ
- อย่ากลัวที่จะลองผิดลองถูก และพยายามทำความเข้าใจหลักการเบื้องหลัง เช่น ทำไม gradient descent ถึงสำคัญ
- เข้าร่วมชุมชนออนไลน์ เช่น Reddit (r/MachineLearning), Stack Overflow หรือ Hugging Face Forums เพื่อแลกเปลี่ยนความรู้และขอความช่วยเหลือ

## ส่วนที่ 1: พื้นฐานที่จำเป็น

*ส่วนนี้เหมาะสำหรับผู้เริ่มต้นที่ต้องการสร้างรากฐานที่มั่นคงในคณิตศาสตร์, Python, โครงข่ายประสาทเทียม และ NLP ก่อนที่จะศึกษาเทคนิคขั้นสูง*

### 1.1 คณิตศาสตร์สำหรับ Machine Learning

| แหล่งข้อมูล                                    | คำอธิบาย                                                                                                   | หัวข้อย่อย (Subtopics)                                                                    | ระดับ (Level)     | ลิงก์                                                                                                                              |
| :------------------------------------------- | :------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- | :---------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| 3Blue1Brown - Linear Algebra                 | ชุดวิดีโอสอนที่ใช้ภาพเคลื่อนไหวอธิบายพีชคณิตเชิงเส้น เหมาะสำหรับเห็นภาพคอนเซปต์ยากๆ เช่น การแปลงเมทริกซ์หรือค่าเจาะจงที่ใช้ใน PCA | เวกเตอร์, เมทริกซ์, การแปลง (transformations), ค่าเจาะจง (eigenvalues - ใช้ใน PCA), เวกเตอร์เจาะจง (eigenvectors - ใช้ใน SVD) | `[Beginner]`      | [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)                                     |
| Khan Academy - Probability and Statistics    | คอร์สออนไลน์ฟรี สอนความน่าจะเป็นและสถิติพื้นฐานที่จำเป็นสำหรับ ML เช่น การแจกแจงที่ใช้ในโมเดลหรือการทดสอบสมมติฐาน | ความน่าจะเป็น, การแจกแจง (distributions), การทดสอบสมมติฐาน, การอนุมานแบบเบย์ (Bayesian inference)      | `[Beginner/Intermediate]` | [Khan Academy Course](https://www.khanacademy.org/math/statistics-probability)                                                     |
| Mathematics for Machine Learning (MML)       | หนังสือเรียนฟรีจาก Cambridge อธิบายคณิตศาสตร์ที่ใช้ใน ML อย่างละเอียด เช่น การหาค่าเหมาะที่สุดสำหรับ gradient descent | พีชคณิตเชิงเส้น, แคลคูลัส, ความน่าจะเป็น, การหาค่าเหมาะที่สุด (optimization)                        | `[Intermediate/Advanced]` | [MML Book](https://mml-book.github.io/)                                                                                       |
| All of Statistics (Wasserman)                | หนังสืออ้างอิงสถิติฉบับสมบูรณ์ เหมาะสำหรับผู้ที่ต้องการเจาะลึก เช่น การวิเคราะห์อนุกรมเวลาใน ML หรือการถดถอยขั้นสูง | การอนุมานทางสถิติ (statistical inference), การทดสอบสมมติฐาน, การถดถอย (regression), อนุกรมเวลา (time series) | `[Advanced]`      | [All of Statistics Book](https://www.stat.cmu.edu/~larry/all-of-statistics/) หรือ [Springer Link](https://link.springer.com/book/10.1007/978-0-387-21736-9) |

### 1.2 การเขียนโปรแกรม Python

| แหล่งข้อมูล                                  | คำอธิบาย                                                                                          | หัวข้อย่อย (Subtopics)                                                                       | ระดับ (Level)        | ลิงก์                                                                                                |
| :------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- | :------------------- | :-------------------------------------------------------------------------------------------------- |
| Python Official Documentation              | เอกสารอ้างอิงอย่างเป็นทางการของ Python อ่านเพื่อเข้าใจพื้นฐานและโมดูลที่ใช้บ่อยใน ML เช่น `random` หรือ `math` | ไวยากรณ์ (syntax), โครงสร้างข้อมูล, โมดูล (modules), ไลบรารีมาตรฐาน (standard library)               | `[All Levels]`   | [Python Docs](https://docs.python.org/3/)                                                               |
| Google's Python Class                      | คอร์สฟรีจาก Google สอน Python พื้นฐาน เหมาะสำหรับผู้เริ่มต้นที่อยากฝึกเขียนโค้ดจริง เช่น การจัดการไฟล์ | พื้นฐาน, สตริง (strings), ลิสต์ (lists), ดิกชันนารี (dictionaries), ไฟล์, regular expressions      | `[Beginner]`       | [Google's Python Class](https://developers.google.com/edu/python/)                                      |
| Corey Schafer - Python Tutorials           | ช่อง YouTube สอน Python เชิงลึก เหมาะสำหรับการเรียนรู้ OOP และงาน data science เช่น การใช้ pandas | พื้นฐาน, การเขียนโปรแกรมเชิงวัตถุ (OOP), การพัฒนาเว็บ, วิทยาศาสตร์ข้อมูล (data science)            | `[Beginner/Intermediate]` | [Corey Schafer YouTube](https://www.youtube.com/c/Coreymschafer)                                        |
| Sentdex - Python Programming Tutorials     | ช่อง YouTube สอน Python แบบประยุกต์ เช่น การสร้างโมเดล ML หรือวิเคราะห์ข้อมูลด้วย NumPy          | Machine Learning, การวิเคราะห์ข้อมูล, การพัฒนาเว็บ, การพัฒนาเกม                                 | `[Intermediate]`    | [Sentdex YouTube](https://www.youtube.com/c/sentdex)                                                 |
| Python for Data Analysis (Wes McKinney)    | หนังสือสอนการใช้ Python (เน้น pandas) สำหรับการวิเคราะห์ข้อมูล เหมาะสำหรับงาน ML และ data science   | pandas, NumPy, การจัดการข้อมูล, การทำความสะอาดข้อมูล, การแสดงภาพข้อมูล (data visualization)     | `[Intermediate/Advanced]` | [Python for Data Analysis](https://wesmckinney.com/book/)                                               |
| Automate the Boring Stuff with Python      | หนังสือและคอร์สฟรี สอน Python พื้นฐานผ่านโปรเจกต์จริง เช่น อัตโนมัติงานเอกสาร เหมาะสำหรับผู้เริ่มต้น   | พื้นฐาน, การจัดการไฟล์, การ scraping เว็บ, การทำงานกับ Excel                                 | `[Beginner]`        | [Automate the Boring Stuff](https://automatetheboringstuff.com/)                                       |

### 1.3 โครงข่ายประสาทเทียม (Neural Networks)

| แหล่งข้อมูล                                    | คำอธิบาย                                                                                                | หัวข้อย่อย (Subtopics)                                                                                   | ระดับ (Level)          | ลิงก์                                                                                                                   |
| :------------------------------------------- | :--------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- | :--------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| DeepLearning.AI - Deep Learning Specialization | ชุดคอร์สออนไลน์บน Coursera อธิบายพื้นฐาน Deep Learning เช่น backpropagation ที่ใช้ในโครงข่ายประสาทเทียม     | โครงข่ายประสาทเทียม, backpropagation, convolutional neural networks (CNNs), recurrent neural networks (RNNs) | `[Beginner/Intermediate]` | [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)                                  |
| 3Blue1Brown - Neural Networks                | ชุดวิดีโอใช้ภาพเคลื่อนไหวอธิบายการทำงานของโครงข่ายประสาทเทียม เช่น gradient descent และเพอร์เซปตรอน       | เพอร์เซปตรอน (perceptrons), ฟังก์ชันกระตุ้น (activation functions), backpropagation, gradient descent   | `[Beginner]`          | [Neural Networks Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)                  |
| fast.ai - Practical Deep Learning for Coders | คอร์สเน้นปฏิบัติ สอน Deep Learning ด้วย fastai (บน PyTorch) เหมาะสำหรับการเริ่มต้นสร้างโมเดลจริง          | การประยุกต์ใช้ Deep Learning, PyTorch, ไลบรารี fastai                                                  | `[Intermediate]`       | [fast.ai Course](https://course.fast.ai/)                                                                            |
| Stanford CS231n - Convolutional Neural Networks | คอร์สจาก Stanford เน้น CNNs สำหรับงาน Computer Vision เช่น การจำแนกภาพ อธิบายสถาปัตยกรรม CNN อย่างละเอียด   | Convolutional layers, pooling layers, สถาปัตยกรรม CNN ยอดนิยม                                       | `[Intermediate/Advanced]` | [CS231n Course](http://cs231n.stanford.edu/)                                                                         |
| Transformers Explained (Hugging Face)        | บทความสั้นจาก Hugging Face อธิบายพื้นฐาน Transformers ซึ่งเป็นรากฐานของ LLM เช่น attention mechanism     | Attention mechanism, self-attention, Transformer architecture                                  | `[Beginner/Intermediate]` | [Transformers Explained](https://huggingface.co/docs/transformers/quicktour)                                         |

### 1.4 การประมวลผลภาษาธรรมชาติ (NLP)

| แหล่งข้อมูล                                                                       | คำอธิบาย                                                                                                  | หัวข้อย่อย (Subtopics)                                                                                                | ระดับ (Level)          | ลิงก์                                                                                                                   |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- | :--------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| Stanford CS224N - NLP with Deep Learning                                       | คอร์สจาก Stanford ครอบคลุม NLP สมัยใหม่ด้วย Deep Learning เช่น การใช้ transformers ในงานแปลภาษา            | Word embeddings, recurrent neural networks (RNNs), transformers, question answering, machine translation  | `[Intermediate/Advanced]` | [CS224N Course](http://web.stanford.edu/class/cs224n/)                                                                   |
| Jurafsky & Martin - Speech and Language Processing (3rd ed. draft)              | หนังสือเรียน NLP ฉบับสมบูรณ์ (ฉบับร่างฟรี) อธิบายตั้งแต่พื้นฐานถึงขั้นสูง เช่น language modeling          | Language modeling, parsing, semantics, discourse, machine translation                                        | `[Intermediate/Advanced]` | [SLP Book (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)                                                       |
| Hugging Face - NLP Course                                                      | คอร์ส interactive สอนการใช้ Transformers library สำหรับงาน NLP เช่น fine-tuning โมเดลสำหรับการจำแนกข้อความ | Transformers, tokenization, fine-tuning, งาน NLP ทั่วไป                                               | `[Beginner/Intermediate]` | [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter1/1)                                                   |
| Natural Language Processing with Python (NLTK Book)                            | หนังสือฟรีสอน NLP ด้วย Python และ NLTK เหมาะสำหรับผู้เริ่มต้นที่อยากฝึกพื้นฐาน เช่น การตัดคำ (tokenization) | Tokenization, part-of-speech tagging, text classification, sentiment analysis                        | `[Beginner]`             | [NLTK Book](https://www.nltk.org/book/)                                                                                 |

### 1.5 แนวทางการเรียนรู้ (How to Proceed)

1. **เริ่มต้นจากพื้นฐาน:** หากคุณเป็นมือใหม่ในวงการ Machine Learning ให้เริ่มด้วยชุดวิดีโอ Linear Algebra ของ 3Blue1Brown และคอร์ส Probability and Statistics ของ Khan Academy เพื่อเข้าใจคณิตศาสตร์พื้นฐาน `[Beginner]`
2. **เรียนรู้ภาษา Python:** ศึกษา Google's Python Class หรือ Automate the Boring Stuff ควบคู่ไปกับการฝึกเขียนโค้ด เช่น สร้างสคริปต์จัดการไฟล์ง่ายๆ `[Beginner/Intermediate]`
3. **ทำความเข้าใจโครงข่ายประสาทเทียม:** เรียน DeepLearning.AI Specialization หรือ fast.ai Course และดูวิดีโอของ 3Blue1Brown เพื่อเห็นภาพการทำงาน ลองสร้างโมเดลจำแนกภาพด้วย PyTorch `[Beginner/Intermediate]`
4. **เจาะลึก NLP:** ศึกษา CS224N ของ Stanford และลองทำ Hugging Face NLP Course เพื่อฝึก fine-tuning โมเดลสำหรับงานจำแนกข้อความ `[Intermediate]`
5. **อ่านหนังสือเพิ่มเติม:** ใช้หนังสือ MML และ Jurafsky & Martin เป็นแหล่งอ้างอิงสำหรับข้อมูลเชิงลึก เช่น การคำนวณ gradient หรือการสร้าง language model `[Intermediate/Advanced]`

**ข้อควรจำ:**
- การเรียนรู้ Machine Learning ต้องใช้เวลาและความสม่ำเสมอ ฝึกทำโปรเจกต์เล็กๆ เช่น จำแนกข้อความหรือวิเคราะห์ข้อมูล เพื่อทดสอบความเข้าใจ
- อย่ากลัวที่จะลองผิดลองถูก และพยายามทำความเข้าใจหลักการเบื้องหลัง เช่น ทำไม gradient descent ถึงสำคัญ
- เข้าร่วมชุมชนออนไลน์ เช่น Reddit (r/MachineLearning), Stack Overflow หรือ Hugging Face Forums เพื่อแลกเปลี่ยนความรู้และขอความช่วยเหลือ

## ส่วนที่ 3: วิศวกร LLM (The LLM Engineer)

*ส่วนนี้เหมาะสำหรับวิศวกรที่ต้องการนำ LLM ไปใช้ในแอปพลิเคชันจริง โดยเน้นการสร้างแอปพลิเคชัน การเพิ่มประสิทธิภาพ และการรักษาความปลอดภัย*

### 3.1 การสร้างแอปพลิเคชันด้วย LLMs (Building Applications with LLMs)

| แหล่งข้อมูล                                  | คำอธิบาย                                                                                      | หัวข้อย่อย (Subtopics)                                                                               | ระดับ (Level)          | ลิงก์                                                                                             |
| :----------------------------------------- | :----------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------- | :--------------------- | :----------------------------------------------------------------------------------------------- |
| LangChain Documentation                    | เอกสารอย่างเป็นทางการของ LangChain เฟรมเวิร์กยอดนิยมสำหรับพัฒนาแอปพลิเคชันที่ขับเคลื่อนด้วย LLM เช่น แชทบอท | LLM chains, agents, memory, document loaders, indexes                                      | `[Intermediate/Advanced]` | [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)                     |
| LlamaIndex Documentation                   | เอกสารอย่างเป็นทางการของ LlamaIndex (เดิมชื่อ GPT Index) เฟรมเวิร์กสำหรับเชื่อม LLM กับข้อมูลภายนอก เช่น ฐานความรู้ | Data connectors, indexes, query engines, data agents                                       | `[Intermediate/Advanced]` | [LlamaIndex Docs](https://docs.llamaindex.ai/en/stable/)                                         |
| Guidance Documentation (Microsoft)         | เอกสารของ Guidance ไลบรารีจาก Microsoft ช่วยควบคุมและจัดการการโต้ตอบกับ LLM ได้ง่ายขึ้น เช่น การสร้างเทมเพลตคำตอบ | Language model control, templating, logic, chat interface                                 | `[Intermediate]`       | [Guidance Docs](https://github.com/guidance-ai/guidance)                                         |
| Building LLM Applications (Hugging Face)   | คอร์สสั้นจาก DeepLearning.AI และ Hugging Face สอนสร้างแอป LLM ด้วยเครื่องมือโอเพนซอร์ส เช่น การทำ retrieval | Prompt engineering, การใช้ LLM โอเพนซอร์ส, retrieval, การสร้าง agents                     | `[Intermediate]`       | [Hugging Face Course](https://www.deeplearning.ai/short-courses/building-llm-applications-with-hugging-face/) |
| Build a GitHub Copilot Clone (YouTube)     | บทแนะนำทีละขั้นตอน สอนสร้างแอปคล้าย GitHub Copilot เช่น การช่วยเขียนโค้ดอัตโนมัติ            | LangChain, OpenAI API, vector databases, Streamlit                                       | `[Intermediate/Advanced]` | [GitHub Copilot Clone Tutorial](https://www.youtube.com/watch?v=M-D2_UrjR-E)                    |
| Build a Custom ChatGPT Clone (YouTube)     | บทแนะนำสอนสร้างแอป ChatGPT แบบกำหนดเอง เช่น ปรับแต่งให้เหมาะกับงานเฉพาะ                    | LangChain, OpenAI API, Gradio                                                    | `[Intermediate/Advanced]` | [ChatGPT Clone Tutorial](https://www.youtube.com/watch?v=RIWbalZ7wTo)                   |

### 3.2 การเพิ่มประสิทธิภาพและความเร็วของ LLM (Enhancing LLM Performance and Efficiency)

| แหล่งข้อมูล                                                           | คำอธิบาย                                                                                                   | หัวข้อย่อย (Subtopics)                                                                                    | ระดับ (Level)          | ลิงก์                                                                                                                     |
| :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------ | :--------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| vLLM Documentation                                                | เอกสารของ vLLM ไลบรารีสำหรับการ inference LLM ที่รวดเร็วและมีประสิทธิภาพ เช่น การใช้ PagedAttention           | PagedAttention, continuous batching, optimized CUDA kernels                                    | `[Advanced]`      | [vLLM Docs](https://docs.vllm.ai/en/latest/)                                                                          |
| Optimum Documentation (Hugging Face)                                 | เอกสารของ Hugging Face Optimum ไลบรารีที่ช่วยเพิ่มประสิทธิภาพโมเดล Transformers เช่น การทำ quantization      | Quantization, pruning, graph optimization, ONNX Runtime, OpenVINO                               | `[Intermediate/Advanced]` | [Optimum Docs](https://huggingface.co/docs/optimum/index)                                                               |
| Text Generation Inference (TGI) (Hugging Face)                          | Inference container ที่พร้อมใช้งานจริงสำหรับ LLM เช่น การรองรับ tensor parallelism                          | Optimized transformers, tensor parallelism, continuous batching, PagedAttention                   | `[Advanced]`      | [TGI (GitHub)](https://github.com/huggingface/text-generation-inference)                                                  |
| FasterTransformer (NVIDIA)                                        | ไลบรารีจาก NVIDIA สำหรับเร่งความเร็วโมเดล Transformer บน GPU เช่น การใช้ INT8 inference                     | Optimized CUDA kernels, INT8 inference, FP16 inference                                          | `[Advanced]`      | [FasterTransformer (GitHub)](https://github.com/NVIDIA/FasterTransformer)                                                |
| DeepSpeed Inference: Enabling Efficient Inference (Microsoft)     | บทความบล็อกที่อธิบายความสามารถด้าน inference ของ DeepSpeed เช่น การลดการใช้หน่วยความจำ                       | Model parallelism, memory optimization, optimized kernels                                      | `[Advanced]`      | [DeepSpeed Inference Blog](https://www.deepspeed.ai/tutorials/inference-tutorial/)                                      |
| Efficient Inference on a Single GPU                               | บทความบล็อกสาธิตการ inference LLM อย่างมีประสิทธิภาพ เช่น การใช้ FlashAttention บน GPU เดียว                | Quantization, FlashAttention, bettertransformer                                                | `[Advanced]`      | [Hugging Face Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)                                         |

### 3.3 ความปลอดภัยของ LLM (LLM Security)

| แหล่งข้อมูล                                             | คำอธิบาย                                                                                                     | หัวข้อย่อย (Subtopics)                                                                                                | ระดับ (Level)   | ลิงก์                                                                                                                                |
| :---------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- | :------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| OWASP Top 10 for LLM Applications                  | รายการความเสี่ยงด้านความปลอดภัย 10 อันดับแรกสำหรับแอป LLM จาก OWASP เช่น การโจมตีด้วย prompt injection          | Prompt injection, insecure output handling, training data poisoning, denial of service, sensitive information disclosure | `[Intermediate/Advanced]` | [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/)                                |
| NIST - Adversarial Machine Learning: A Taxonomy and Terminology    | เอกสารจาก NIST อธิบายคำจำกัดความและการโจมตีด้าน Adversarial ML เช่น การโจมตีแบบ poisoning                     | Poisoning attacks, evasion attacks, model extraction                                                  | `[Advanced]`     | [NIST Adversarial ML](https://csrc.nist.gov/pubs/ai/100/2/final)                                                                 |
| Robustness Gym (Hugging Face)                         | เครื่องมือประเมินความทนทานของโมเดล NLP เช่น การทดสอบการโจมตีแบบ adversarial                                   | Attacks, evaluation metrics, robustness analysis                                              | `[Intermediate/Advanced]` | [Robustness Gym (GitHub)](https://github.com/robustness-gym/robustness-gym)                                              |
| Garak (llm-security.github.io)                        | เครื่องมือสแกนช่องโหว่ของ LLM เช่น การตรวจจับ prompt injection หรือ data leakage                           | Prompt injection, data leakage, jailbreaking                                                  | `[Intermediate/Advanced]` | [Garak](https://llm-security.github.io/garak/)                                                          |

### 3.4 แนวทางการเรียนรู้ (How to Proceed)

1. **สร้างแอปพลิเคชันพื้นฐาน:** เริ่มด้วยการสร้างแอปง่ายๆ โดยใช้ LangChain หรือ LlamaIndex เช่น แชทบอทตอบคำถามจากเอกสาร `[Intermediate]`
2. **ทดลองกับ Guidance:** เรียนรู้วิธีควบคุม LLM ให้แม่นยำขึ้นด้วย Guidance เช่น การกำหนดรูปแบบคำตอบ `[Intermediate]`
3. **ศึกษาการเพิ่มประสิทธิภาพ:** ทำความเข้าใจเทคนิคต่างๆ เช่น quantization, pruning และ optimized kernels โดยใช้ vLLM, Optimum หรือ TGI `[Advanced]`
4. **ให้ความสำคัญกับความปลอดภัย:** ศึกษา OWASP Top 10 for LLM และ NIST Adversarial ML เพื่อเข้าใจความเสี่ยงและวิธีป้องกัน `[Intermediate/Advanced]`
5. **ทดลองใช้เครื่องมือ:** ลองใช้ Robustness Gym และ Garak เพื่อประเมินและปรับปรุงความทนทานและความปลอดภัยของโมเดล `[Intermediate/Advanced]`

**ข้อควรจำ:**
- การสร้างแอป LLM ต้องคำนึงถึงทั้งประสิทธิภาพและความปลอดภัย
- ทดลอง deploy โมเดลบนสภาพแวดล้อมจริง เช่น คลาวด์ (AWS, GCP) เพื่อฝึกการใช้งาน
- เข้าร่วมชุมชน เช่น Hugging Face Forums หรือ Reddit (r/MachineLearning) เพื่ออัปเดตเทคนิคใหม่ๆ

## ส่วนที่ 4: สคริปต์และ Repository ขั้นสูง (Advanced Scripts & Repositories)

*ส่วนนี้รวบรวมสคริปต์ repositories และแหล่งข้อมูลขั้นสูงที่เป็นประโยชน์สำหรับผู้ที่ต้องการศึกษาและพัฒนา LLM ในเชิงลึก*

### 4.1 DeepSeek AI Repositories

*Repositories จาก DeepSeek AI บริษัทวิจัย AI ที่เน้นงาน Open Source*

#### Performance Optimization

| Repository                                  | คำอธิบาย                                                                                                       | ลิงก์                                                                                    |
| :------------------------------------------ | :------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------- |
| DeepSpeed-FastGen                          | ระบบ inference รวดเร็วและมีประสิทธิภาพ สร้างบน DeepSpeed เหมาะสำหรับงาน real-time                          | [DeepSpeed-FastGen (GitHub)](https://github.com/DeepSpeed-MII/DeepSpeed-FastGen)             |
| DeepSpeed-Kernels                           | Custom CUDA kernels สำหรับ DeepSpeed เพิ่มความเร็วในการคำนวณโมเดล LLM                                     | [DeepSpeed-Kernels (GitHub)](https://github.com/microsoft/DeepSpeed-Kernels)                |
| DeepSpeed-MII                            | ไลบรารีสำหรับ deploy และเพิ่มประสิทธิภาพ LLM ด้วย DeepSpeed เช่น ลดการใช้หน่วยความจำ                          | [DeepSpeed-MII (GitHub)](https://github.com/microsoft/DeepSpeed-MII)                           |
| DeepSpeed                                     | ไลบรารีจาก Microsoft สำหรับ optimize การฝึกและ inference โมเดลขนาดใหญ่                                   | [DeepSpeed (GitHub)](https://github.com/microsoft/DeepSpeed)                                  |
| 3FS: Fast, Flexible, and Friendly Sparse Attention    | การใช้งาน Sparse Attention เพื่อเพิ่มประสิทธิภาพการคำนวณ attention                                     | [3FS](https://github.com/DeepSeek-AI/3FS)             |
| DeepGEMM                                  | GEMM (General Matrix Multiplication) ที่มีประสิทธิภาพสูง ใช้ในงานคำนวณเชิงลึก                            | [DeepGEMM (GitHub)](https://github.com/DeepSeek-AI/DeepGEMM)                                |

#### Model Architectures

| Repository                                       | คำอธิบาย                                                                  | ลิงก์                                                                                           |
| :----------------------------------------------- | :------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------- |
| DeepSeek-LLM                                  | โครงการโอเพนซอร์สสำหรับ DeepSeek LLM โมเดลภาษาขนาดใหญ่                     | [DeepSeek-LLM (GitHub)](https://github.com/deepseek-ai/DeepSeek-LLM)                         |
| DeepSeek-MoE                                   | โครงการโอเพนซอร์สสำหรับ DeepSeek MoE โมเดลแบบ mixture-of-experts           | [DeepSeek-MoE (GitHub)](https://github.com/deepseek-ai/DeepSeek-MoE)                        |
| FastMoE                                        | การใช้งาน Mixture of Experts (MoE) ที่รวดเร็ว เหมาะสำหรับโมเดลขนาดใหญ่       | [FastMoE](https://github.com/laekov/fastmoe)                                         |

#### Datasets
| Repository                                       | คำอธิบาย                                                                  | ลิงก์                                                                                           |
| :----------------------------------------------- | :------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------- |
| UltraFeedback                                  | ชุดข้อมูลขนาดใหญ่สำหรับ preference data ใช้ฝึกโมเดลให้สอดคล้องกับมนุษย์      | [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)                     |

#### Other Tools
| Repository                                  | คำอธิบาย                                                                       | ลิงก์                                                                                       |
| :----------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| ChatTemplate                              | เครื่องมือเพิ่มความสามารถด้านการสนทนาให้โมเดล เช่น การสร้างเทมเพลตแชท            | [ChatTemplate](https://github.com/DeepSeek-AI/ChatTemplate)                             |

### 4.2 Uncensored AI Models

*ส่วนนี้เกี่ยวกับโมเดล AI ที่ไม่มีการเซ็นเซอร์ (uncensored) ซึ่งอาจมีเนื้อหาที่ไม่เหมาะสมหรือไม่ปลอดภัย ควรใช้ด้วยความระมัดระวังและรับผิดชอบต่อผลที่ตามมา*

**คำเตือน:** การใช้ uncensored models อาจนำไปสู่การสร้างเนื้อหาที่เป็นอันตราย ไม่เหมาะสม หรือผิดกฎหมาย โปรดใช้อย่างมีสติ

| Repository/Model                | คำอธิบาย                                                                                                 | ลิงก์                                                                                                  |
| :------------------------------ | :-------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| Dolphin (Eric Hartford)          | โมเดล fine-tuned จาก Llama 2 และ WizardLM เน้นลด bias และ censorship เหมาะสำหรับงานวิจัย                 | [Dolphin (Hugging Face)](https://huggingface.co/ehartford/dolphin-2.2.1-mistral-7b)                   |
| OpenHathi-7B-Hi-v0.1-Base (Sarvam AI) | โมเดลสำหรับภาษาฮินดี อังกฤษ และฮิงลิช พัฒนาโดย Sarvam AI ไม่ใช่ภาษาไทย                                | [OpenHathi (Hugging Face)](https://huggingface.co/sarvamai/OpenHathi-7B-Hi-v0.1-Base)                |
| OpenOrca (OpenOrca)              | ชุดข้อมูลและโมเดลโอเพนซอร์ส เน้นการตามคำสั่ง (instruction following)                                     | [OpenOrca (Hugging Face)](https://huggingface.co/Open-Orca)                                          |
| mpt-30b-chat (MosaicML)         | โมเดลสำหรับการสนทนา ขนาด 30B parameters เหมาะสำหรับงานแชทที่ต้องการความยืดหยุ่น                        | [mpt-30b-chat (Hugging Face)](https://huggingface.co/mosaicml/mpt-30b-chat)                         |
| WizardLM (Microsoft)             | โมเดล fine-tuned จาก LLaMA ด้วยชุดข้อมูลอัตโนมัติ เน้นความสามารถในการตอบคำถามที่ซับซ้อน                | [WizardLM (GitHub)](https://github.com/nlpxucan/WizardLM)                                          |
| OpenThaiGPT (OpenThaiGPT)        | โมเดลโอเพนซอร์สสำหรับภาษาไทย พัฒนาโดยชุมชนไทย เหมาะสำหรับงาน NLP ภาษาไทย                               | [OpenThaiGPT (Hugging Face)](https://huggingface.co/openthaigpt)                                   |

### 4.3 Fine-Tuning Tables

| Repository                                          | คำอธิบาย                                                                                                          | ลิงก์                                                                                                             |
| :-------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| Unsloth                                            | เครื่องมือสำหรับ fine-tuning และ quantization ที่รวดเร็วและประหยัดหน่วยความจำ เช่น รันโมเดล 7B บน GPU เดียว           | [Unsloth (GitHub)](https://github.com/unslothai/unsloth)                                                            |
| LLaMA-Factory                                        | เครื่องมือสำหรับ fine-tuning LLaMA และโมเดลอื่นๆ รองรับการปรับแต่งหลากหลายรูปแบบ                                     | [LLaMA-Factory (GitHub)](https://github.com/hiyouga/LLaMA-Factory)                                               |
| Axolotl                                             | เครื่องมือ fine-tuning LLM ที่ยืดหยุ่น เหมาะสำหรับงานวิจัยและการทดลอง                                            | [Axolotl (GitHub)](https://github.com/OpenAccess-AI-Collective/axolotl)                                        |
| PEFT (Hugging Face)                                  | วิธี Parameter-Efficient Fine-Tuning (PEFT) จาก Hugging Face เช่น LoRA และ Prompt Tuning                        | [PEFT (GitHub)](https://github.com/huggingface/peft)                                                        |
| trlX                                              | ไลบรารีสำหรับฝึก LLM ด้วย reinforcement learning (fork จาก TRL) เช่น การใช้ PPO                                 | [trlX](https://github.com/CarperAI/trlx)                                                                  |
| AutoTrain (Hugging Face)                           | เครื่องมือ fine-tuning โมเดลโดยไม่ต้องเขียนโค้ด เหมาะสำหรับผู้เริ่มต้นและงานด่วน                                  | [AutoTrain (Hugging Face)](https://huggingface.co/autotrain)                                              |

### 4.4 Awesome Colab Notebooks

| Notebook                                           | คำอธิบาย                                                                                                      | ลิงก์                                                                                                                |
| :------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| Kaggle Notebooks - LLM Science Exam              | ตัวอย่าง notebooks จาก Kaggle สอนใช้ LLM ตอบคำถามวิทยาศาสตร์ เช่น การประมวลผลข้อสอบ                             | [Kaggle LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam/code)                          |
| ML-Engineering Open Book                         | ชุด notebooks และทรัพยากรสำหรับวิศวกรรม ML เช่น การ deploy โมเดล                                                | [ml-engineering-open-book](https://github.com/stas00/ml-engineering-open-book/tree/master/open-book)                |
| MLOps Course (Made With ML)                      | คอร์สที่มี notebooks ปฏิบัติ เช่น การจัดการ pipeline ML                                                        | [mlops-course](https://github.com/GokuMohandas/mlops-course)                                                       |
| Unsloth Fine-Tuning Example                      | Notebook ตัวอย่างจาก Unsloth สอน fine-tuning โมเดล เช่น LLaMA บน Colab                                         | [Unsloth Colab](https://colab.research.google.com/drive/1zB1t1qFO5n4bdnqP4KauS2sV_lPtP2Q)                        |

### 4.5 Research Papers

#### Quantization
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models: [SmoothQuant](https://arxiv.org/abs/2211.10438)
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers: [GPTQ](https://arxiv.org/abs/2210.17323)
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration: [AWQ](https://arxiv.org/abs/2306.00978)
- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers: [ZeroQuant](https://arxiv.org/abs/2206.01861)

#### Fine-Tuning Techniques
- LoRA: Low-Rank Adaptation of Large Language Models: [LoRA](https://arxiv.org/abs/2106.09685)
- QLoRA: Efficient Finetuning of Quantized LLMs: [QLoRA](https://arxiv.org/abs/2305.14314)
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model: [DPO](https://arxiv.org/abs/2305.18290)
- ORPO: Monolithic Preference Optimization without Reference Model: [ORPO](https://arxiv.org/abs/2403.07687)

#### Distributed Training
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models: [ZeRO](https://arxiv.org/abs/1910.02054)
- Megatron-LM: Training Multi-Billion Parameter Models Using Model Parallelism: [Megatron-LM](https://arxiv.org/abs/1909.08053)
- Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B: [Megatron-Turing NLG 530B](https://arxiv.org/abs/2201.11990)

#### Other Topics
- Llama 2: Open Foundation and Fine-Tuned Chat Models: [Llama2](https://arxiv.org/abs/2307.09288)
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness: [FlashAttention](https://arxiv.org/abs/2205.14135)
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning: [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- A Survey of Large Language Models: [LLM Survey](https://arxiv.org/abs/2303.18223)
- Instruction Tuning for Large Language Models: A Survey: [Instruction Tuning](https://arxiv.org/abs/2308.10792)

### 4.6 Additional Learning Resources

#### Project Ideas and Datasets
- [Kaggle](https://www.kaggle.com/): แพลตฟอร์มแข่งขัน Data Science และชุดข้อมูลมากมาย
- [NLP Datasets (Hugging Face)](https://huggingface.co/datasets): ชุดข้อมูลสำหรับงาน NLP

#### Online Courses
- Andrew Ng's Courses on Coursera and DeepLearning.AI: คอร์ส Machine Learning และ Deep Learning ชั้นนำ
- fast.ai: คอร์ส Deep Learning เน้นปฏิบัติจริง

#### Podcasts
- Lex Fridman Podcast: สัมภาษณ์บุคคลสำคัญในวงการ AI และวิทยาศาสตร์
- Data Skeptic: พูดถึง Data Science, Machine Learning และสถิติ
- The TWIML AI Podcast: อัปเดตข่าวสารและพัฒนาการในวงการ ML และ AI

#### YouTube Channels
- Sentdex: สอน Python และ Machine Learning
- Corey Schafer: สอน Python ทุกระดับ
- 3Blue1Brown: อธิบายคณิตศาสตร์ด้วยภาพเคลื่อนไหว
- Two Minute Papers: นำเสนอ paper วิจัย AI แบบสั้น

#### Online Communities
- Reddit: r/MachineLearning, r/LanguageTechnology
- Stack Overflow: ถาม-ตอบสำหรับโปรแกรมเมอร์
- Hugging Face Forums: ฟอรัมสำหรับผู้ใช้ Hugging Face
- Discord Servers: ค้นหา "LLM Discord" หรือ "AI Discord" เพื่อชุมชนที่เน้น AI และ LLM

## ส่วนที่ 5: การประยุกต์ใช้ LLM ในโลกจริงและเทรนด์อนาคต (Applying LLMs in Real-World and Future Trends)

*ส่วนนี้เน้นการนำ LLM ไปใช้ในโปรเจกต์จริง การวัดผล และการติดตามเทรนด์ใหม่ๆ ในวงการ เพื่อให้พร้อมสำหรับการพัฒนาในอนาคต*

### 5.1 โปรเจกต์ตัวอย่างที่ใช้ LLM (Real-World LLM Projects)

| โปรเจกต์                                      | คำอธิบาย                                                                                       | เทคโนโลยีที่ใช้                              | ระดับ (Level)          | ลิงก์                                                                                             |
| :------------------------------------------- | :----------------------------------------------------------------------------------------------- | :------------------------------------------ | :--------------------- | :----------------------------------------------------------------------------------------------- |
| Chatbot for Customer Support                 | แชทบอทช่วยตอบคำถามลูกค้า เช่น การแนะนำสินค้าหรือแก้ปัญหาเบื้องต้น                              | LangChain, Llama 3, Gradio                  | `[Intermediate]`       | [Example Repo](https://github.com/langchain-ai/langchain/tree/master/templates)                  |
| Document Summarization Tool                  | เครื่องมือสรุปเอกสารยาว เช่น รายงานหรือบทความวิชาการ                                          | Hugging Face Transformers, Mistral 7B       | `[Intermediate/Advanced]` | [Hugging Face Tutorial](https://huggingface.co/docs/transformers/tasks/summarization)           |
| Code Generation Assistant                    | ผู้ช่วยเขียนโค้ด เช่น การสร้างฟังก์ชัน Python หรือแก้ bug                                      | OpenAI API, GitHub Copilot Clone, Streamlit | `[Intermediate/Advanced]` | [Copilot Clone Tutorial](https://www.youtube.com/watch?v=M-D2_UrjR-E)                          |
| Multilingual Translator                      | ระบบแปลภาษาหลายภาษา เช่น ไทย-อังกฤษ-จีน โดยใช้โมเดล open-source                              | OpenThaiGPT, mBART                          | `[Advanced]`           | [mBART Docs](https://huggingface.co/docs/transformers/model_doc/mbart)                         |

### 5.2 การวัดผลและปรับปรุง LLM (Evaluation and Improvement)

| แหล่งข้อมูล                                    | คำอธิบาย                                                                                           | หัวข้อย่อย (Subtopics)                           | ระดับ (Level)          | ลิงก์                                                                                             |
| :------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :---------------------------------------------- | :--------------------- | :----------------------------------------------------------------------------------------------- |
| Hugging Face Evaluate                        | ไลบรารีสำหรับวัดผลโมเดล เช่น ความแม่นยำ (accuracy) หรือ BLEU score ในงานแปลภาษา                   | Metrics, evaluation datasets, benchmarking      | `[Intermediate]`       | [Evaluate Docs](https://huggingface.co/docs/evaluate/index)                                      |
| EleutherAI LM Evaluation Harness             | เครื่องมือวัดประสิทธิภาพ LLM ด้วยชุดข้อมูลมาตรฐาน เช่น MMLU หรือ TruthfulQA                       | Task-specific evaluation, zero-shot testing     | `[Advanced]`           | [LM Eval (GitHub)](https://github.com/EleutherAI/lm-evaluation-harness)                         |
| HumanEval: Evaluating Large Language Models  | Paper อธิบาย HumanEval ชุดข้อมูลสำหรับทดสอบความสามารถเขียนโค้ดของ LLM                             | Code generation metrics, functional correctness | `[Advanced]`           | [HumanEval (Arxiv)](https://arxiv.org/abs/2107.03374)                                           |
| MT-Bench: A Multi-turn Benchmark            | Paper นำเสนอ MT-Bench ชุดทดสอบสำหรับประเมินการสนทนาหลายรอบของ LLM                                | Multi-turn dialogue, human-like responses       | `[Advanced]`           | [MT-Bench (Arxiv)](https://arxiv.org/abs/2306.05685)                                            |

### 5.3 เทรนด์และนวัตกรรมใหม่ในวงการ LLM (Emerging Trends and Innovations)

| หัวข้อ                                         | คำอธิบาย                                                                                           | แหล่งข้อมูลที่เกี่ยวข้อง                        | ลิงก์                                                                                             |
| :------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------------------------------------------------- |
| Multimodal LLMs                              | โมเดลที่รวมข้อความ รูปภาพ และข้อมูลอื่นๆ เช่น CLIP-ViT หรือ DALL-E 3                                | Paper: "Flamingo" (Arxiv)                      | [Flamingo (Arxiv)](https://arxiv.org/abs/2204.14198)                                            |
| Smaller, Efficient Models                    | โมเดลขนาดเล็กแต่ทรงพลัง เช่น Phi-3 (Microsoft) หรือ TinyLlama เหมาะสำหรับอุปกรณ์จำกัดทรัพยากร       | Phi-3 Blog (Microsoft)                         | [Phi-3 Blog](https://azure.microsoft.com/en-us/blog/introducing-phi-3/)                         |
| AI Agents and Tool Integration               | การพัฒนา AI agents ที่ใช้ LLM ร่วมกับเครื่องมือ เช่น AutoGen หรือ AgentGPT                          | AutoGen (GitHub)                               | [AutoGen (GitHub)](https://github.com/microsoft/autogen)                                        |
| Ethical AI and Regulation                    | แนวโน้มด้านจริยธรรมและกฎระเบียบ เช่น การลด bias หรือการควบคุมเนื้อหาที่สร้างโดย LLM                 | AI Ethics Guidelines (UNESCO)                  | [UNESCO AI Ethics](https://unesdoc.unesco.org/ark:/48223/pf0000381137)                          |

### 5.4 แนวทางการเรียนรู้ (How to Proceed)

1. **เริ่มต้นด้วยโปรเจกต์ง่ายๆ:** ลองสร้างแอปพื้นฐาน เช่น แชทบอทหรือเครื่องมือสรุปข้อความ โดยใช้ LangChain หรือ Hugging Face `[Intermediate]`
2. **วัดผลโมเดล:** ใช้เครื่องมืออย่าง Hugging Face Evaluate หรือ LM Evaluation Harness เพื่อทดสอบประสิทธิภาพโมเดลที่พัฒนา `[Intermediate/Advanced]`
3. **สำรวจเทรนด์ใหม่:** อ่าน paper เช่น Flamingo หรือติดตามบล็อกจาก Microsoft และ xAI เพื่ออัปเดตเทคโนโลยี เช่น multimodal LLMs `[Advanced]`
4. **พัฒนาโปรเจกต์ขั้นสูง:** ลองรวม LLM กับเครื่องมืออื่น เช่น สร้าง AI agent ด้วย AutoGen หรือโมเดลขนาดเล็กสำหรับ edge devices `[Advanced]`
5. **คำนึงถึงจริยธรรม:** ตรวจสอบ bias และความปลอดภัยของโมเดล โดยอิงจากแนวทาง เช่น UNESCO AI Ethics `[All Levels]`

**ข้อควรจำ:**
- การนำ LLM ไปใช้จริงต้องทดสอบในสภาพแวดล้อมที่หลากหลาย เช่น บนคลาวด์หรืออุปกรณ์ท้องถิ่น
- ติดตามงานวิจัยและชุมชน เช่น Arxiv หรือ X (Twitter) เพื่อไม่พลาดนวัตกรรมล่าสุด
- ความรับผิดชอบต่อผลกระทบของ LLM เป็นสิ่งสำคัญ เช่น การหลีกเลี่ยงเนื้อหาที่ไม่เหมาะสม

## ส่วนที่ 6: ศัพท์ที่ต้องรู้ในวงการ LLM (Key Terminology in the LLM Field)

*ส่วนนี้รวบรวมคำศัพท์สำคัญที่ใช้ในวงการ Large Language Models (LLM) พร้อมคำอธิบาย เพื่อช่วยให้เข้าใจแนวคิดและเทคนิคต่างๆ*

### 6.1 ศัพท์พื้นฐาน (Basic Terminology)

| คำศัพท์                   | ความหมาย                                                                                       |
| :----------------------- | :--------------------------------------------------------------------------------------------- |
| LLM (Large Language Model) | โมเดลภาษาขนาดใหญ่ที่ฝึกด้วยข้อมูลข้อความจำนวนมาก ใช้สร้างข้อความหรือตอบคำถาม เช่น GPT หรือ LLaMA |
| Transformer              | สถาปัตยกรรมพื้นฐานของ LLM ใช้กลไก attention เพื่อประมวลผลข้อความแบบลำดับ                       |
| Attention                | กลไกที่ช่วยโมเดลโฟกัสส่วนสำคัญของข้อความ เช่น คำที่เกี่ยวข้องกันในประโยค                       |
| Pre-training             | การฝึกโมเดลด้วยข้อมูลขนาดใหญ่ก่อน เพื่อให้เข้าใจภาษาทั่วไป เช่น การทำนายคำถัดไป                |
| Fine-tuning             | การปรับโมเดลที่ pre-trained แล้วให้เหมาะกับงานเฉพาะ เช่น การตอบคำถามในโดเมนหนึ่ง              |
| Token                    | หน่วยย่อยของข้อความ เช่น คำหรือส่วนของคำ ที่โมเดลใช้ในการประมวลผล                            |
| Embedding                | การแปลงคำหรือ token เป็นเวกเตอร์ตัวเลข เพื่อให้โมเดลเข้าใจความหมายและความสัมพันธ์              |

### 6.2 ศัพท์เกี่ยวกับเทคนิคและวิธีการ (Techniques and Methods)

| คำศัพท์                   | ความหมาย                                                                                       |
| :----------------------- | :--------------------------------------------------------------------------------------------- |
| Self-Attention           | กลไก attention ที่คำนวณความสัมพันธ์ระหว่างทุก token ในประโยคเดียวกัน                           |
| Multi-Head Attention     | การใช้ attention หลายชั้นใน Transformer เพื่อจับความสัมพันธ์ที่หลากหลาย                      |
| Masked Language Model (MLM) | วิธี pre-training ที่ซ่อนบางคำในประโยคให้โมเดลทาย เช่น ที่ใช้ใน BERT                        |
| Autoregressive Model     | โมเดลที่สร้างข้อความโดยทำนายคำถัดไปจากคำก่อนหน้า เช่น GPT                                   |
| LoRA (Low-Rank Adaptation) | เทคนิค fine-tuning ที่ปรับพารามิเตอร์เพียงบางส่วน เพื่อประหยัดทรัพยากร                      |
| QLoRA                    | การรวม quantization กับ LoRA เพื่อ fine-tuning ที่ใช้หน่วยความจำน้อยลง                       |
| Quantization             | การลดขนาดโมเดลโดยเปลี่ยนน้ำหนักจาก FP32 เป็น INT8 หรือ 4-bit เพื่อให้รันได้เร็วขึ้น           |
| Prompt Engineering       | การออกแบบคำสั่งหรือคำถามให้ LLM ตอบได้ดีที่สุด เช่น การใช้ตัวอย่างใน prompt                   |
| RLHF (Reinforcement Learning from Human Feedback) | การฝึกโมเดลด้วย feedback จากมนุษย์ เพื่อให้ตอบได้สอดคล้องกับความต้องการ                     |

### 6.3 ศัพท์เกี่ยวกับประสิทธิภาพและการใช้งาน (Performance and Deployment)

| คำศัพท์                   | ความหมาย                                                                                       |
| :----------------------- | :--------------------------------------------------------------------------------------------- |
| Inference                | การใช้โมเดลที่ฝึกแล้วเพื่อสร้างคำตอบหรือทำนายผล เช่น การตอบคำถาม                            |
| Latency                  | เวลาที่โมเดลใช้ในการประมวลผลและให้คำตอบ                                                  |
| Throughput               | จำนวนคำขอที่โมเดลจัดการได้ในหนึ่งหน่วยเวลา เช่น คำต่อวินาที                                |
| Model Parallelism        | การแบ่งโมเดลไปรันบน GPU หลายตัว เพื่อจัดการโมเดลขนาดใหญ่                                  |
| Data Parallelism         | การแบ่งข้อมูลไปฝึกบน GPU หลายตัว เพื่อเร่งการฝึกโมเดล                                    |
| PagedAttention           | เทคนิคจัดการหน่วยความจำใน inference เพื่อให้รันโมเดลใหญ่ได้อย่างมีประสิทธิภาพ              |
| FlashAttention           | เทคนิค attention ที่เร็วและประหยัดหน่วยความจำ ใช้ใน pre-training และ inference             |

### 6.4 ศัพท์เกี่ยวกับความปลอดภัยและจริยธรรม (Safety and Ethics)

| คำศัพท์                   | ความหมาย                                                                                       |
| :----------------------- | :--------------------------------------------------------------------------------------------- |
| Bias                     | อคติในโมเดลที่เกิดจากข้อมูลฝึก เช่น การตอบที่ลำเอียงต่อเพศหรือเชื้อชาติ                    |
| Prompt Injection         | การโจมตีโดยใส่คำสั่งอันตรายใน prompt เพื่อหลอกให้โมเดลทำสิ่งที่ไม่ควร                       |
| Jailbreaking             | การหลบเลี่ยงข้อจำกัดของโมเดลเพื่อให้ตอบคำถามที่ถูกห้าม เช่น คำถามที่ผิดกฎหมาย               |
| Data Poisoning           | การปนเปื้อนข้อมูลฝึกด้วยข้อมูลที่เป็นอันตราย เพื่อให้โมเดลทำงานผิดพลาด                      |
| Explainability           | ความสามารถในการอธิบายว่าโมเดลตัดสินใจหรือตอบคำถามอย่างไร                                 |

### 6.5 ศัพท์เกี่ยวกับการวัดผลและชุดข้อมูล (Evaluation and Datasets)

| คำศัพท์                   | ความหมาย                                                                                       |
| :----------------------- | :--------------------------------------------------------------------------------------------- |
| BLEU Score               | ตัววัดความแม่นยำของการแปลภาษา โดยเปรียบเทียบกับคำแปลจากมนุษย์                            |
| Perplexity               | ตัววัดความยากที่โมเดลเจอในการทำนายคำถัดไป ค่ายิ่งต่ำยิ่งดี                                |
| MMLU (Massive Multitask Language Understanding) | ชุดทดสอบความรู้ทั่วไปของ LLM ครอบคลุมหลายสาขา เช่น คณิตศาสตร์และวิทยาศาสตร์                |
| TruthfulQA               | ชุดทดสอบที่วัดความถูกต้องและความน่าเชื่อถือของคำตอบจาก LLM                                 |
| HumanEval                | ชุดข้อมูลสำหรับทดสอบความสามารถเขียนโค้ดของ LLM เช่น การแก้โจทย์โปรแกรม                    |

### 6.6 แนวทางการเรียนรู้ศัพท์ (How to Proceed)

1. **เริ่มจากพื้นฐาน:** ทำความเข้าใจคำศัพท์พื้นฐาน เช่น LLM, Transformer, และ Token ก่อน เพื่อสร้างรากฐาน `[Beginner]`
2. **เชื่อมโยงกับเทคนิค:** เรียนรู้ศัพท์อย่าง LoRA หรือ Quantization ควบคู่กับการอ่าน paper หรือทดลองใช้เครื่องมือ `[Intermediate]`
3. **ฝึกใช้จริง:** ลองใช้คำศัพท์ในโปรเจกต์ เช่น วัด latency หรือทำ prompt engineering เพื่อเห็นความหมายชัดเจน `[Intermediate/Advanced]`
4. **ติดตามความปลอดภัย:** ศึกษา bias และ prompt injection เพื่อเข้าใจความท้าทายด้านจริยธรรม `[All Levels]`
5. **อัปเดตคำศัพท์ใหม่:** ติดตามบล็อกหรือ paper เช่น จาก Hugging Face หรือ Arxiv เพื่อรู้จักคำศัพท์ล่าสุด `[Advanced]`

**ข้อควรจำ:**
- คำศัพท์ในวงการ LLM อาจเปลี่ยนแปลงหรือมีคำใหม่เพิ่มขึ้นตามเทคโนโลยีที่พัฒนา
- การเข้าใจศัพท์จะช่วยให้อ่านเอกสารวิจัยหรือใช้งานเครื่องมือได้ดีขึ้น
- ลองจดคำศัพท์ที่เจอบ่อยและทบทวนเป็นระยะเพื่อความคุ้นเคย