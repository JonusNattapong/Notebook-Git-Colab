import pandas as pd
import os

def create_fundamentals_data():
    data = {
        'section': [
            'Mathematics',
            'Mathematics', 
            'Python',
            'Python',
            'Neural Networks',
            'Neural Networks',
            'NLP',
            'NLP'
        ],
        'topic': [
            'Linear Algebra',
            'Statistics',
            'Core Python',
            'ML Libraries',
            'Fundamentals',
            'Training',
            'Text Processing',
            'Advanced NLP'
        ],
        'description': [
            'Vectors, matrices, calculus essentials for ML',
            'Probability, distributions, hypothesis testing',
            'Variables, functions, data structures',
            'NumPy, Pandas, Matplotlib, TensorFlow',
            'Neural network architecture and components',
            'Loss functions, optimizers, regularization',
            'Tokenization, embeddings, preprocessing',
            'Transformers, attention mechanisms'
        ],
        'resource': [
            'Mathematics for Machine Learning (Cambridge)',
            'Statistics and Probability (Khan Academy)',
            'Python for Data Science (Coursera)',
            'Python ML Libraries Guide (RealPython)',
            'Neural Networks from Scratch (Manning)',
            'Deep Learning Specialization (Coursera)',
            'NLP with Python (O\'Reilly)',
            'Transformers Guide (Hugging Face)'
        ],
        'url': [
            'https://mml-book.github.io/',
            'https://www.khanacademy.org/math/statistics-probability',
            'https://www.coursera.org/python-data-science',
            'https://realpython.com/learning-paths/data-science-python/',
            'https://www.manning.com/books/neural-networks-from-scratch',
            'https://www.coursera.org/specializations/deep-learning',
            'https://www.oreilly.com/library/nlp-python/',
            'https://huggingface.co/course/chapter1/1'
        ]
    }
    return pd.DataFrame(data)

def create_notebooks_data():
    data = {
        'category': [
            'Basics',
            'Basics',
            'Advanced',
            'Advanced',
            'Projects',
            'Projects'
        ],
        'title': [
            'ML Fundamentals',
            'Deep Learning 101',
            'LLM Fine-tuning',
            'Transformer Models',
            'NLP Applications',
            'Computer Vision'
        ],
        'platform': [
            'Google Colab',
            'Kaggle',
            'Google Colab',
            'Kaggle',
            'Google Colab',
            'Kaggle'
        ],
        'description': [
            'Introduction to machine learning concepts',
            'Deep learning fundamentals with PyTorch',
            'Fine-tuning LLMs with minimal resources',
            'Understanding transformer architectures',
            'Building NLP applications with LLMs',
            'Computer vision projects and applications'
        ],
        'link': [
            'https://colab.research.google.com/ml-fundamentals',
            'https://kaggle.com/notebooks/deep-learning-101',
            'https://colab.research.google.com/llm-fine-tuning',
            'https://kaggle.com/notebooks/transformer-models',
            'https://colab.research.google.com/nlp-applications',
            'https://kaggle.com/notebooks/computer-vision'
        ]
    }
    return pd.DataFrame(data)

def create_courses_data():
    data = {
        'level': [
            'Beginner',
            'Intermediate',
            'Advanced',
            'Beginner',
            'Intermediate',
            'Advanced'
        ],
        'name': [
            'ML Crash Course',
            'Deep Learning Specialization',
            'Advanced AI Engineering',
            'Python for AI',
            'NLP with Transformers',
            'LLM Systems Design'
        ],
        'provider': [
            'Google',
            'Coursera',
            'Stanford',
            'Microsoft',
            'Hugging Face',
            'MIT'
        ],
        'duration': [
            '15 hours',
            '5 months',
            '12 weeks',
            '20 hours',
            '3 months',
            '16 weeks'
        ],
        'url': [
            'https://developers.google.com/machine-learning/crash-course',
            'https://www.coursera.org/specializations/deep-learning',
            'https://stanford.edu/cs234',
            'https://microsoft.com/learn/ai-fundamentals',
            'https://huggingface.co/course',
            'https://mit.edu/llm-systems'
        ]
    }
    return pd.DataFrame(data)

def create_tools_data():
    data = {
        'category': [
            'Framework',
            'Framework',
            'Library',
            'Library',
            'Tool',
            'Tool'
        ],
        'name': [
            'TensorFlow',
            'PyTorch',
            'Hugging Face',
            'Scikit-learn',
            'Weights & Biases',
            'Ray'
        ],
        'use_case': [
            'ML/DL Development',
            'Research/Prototyping',
            'NLP/Transformers',
            'Classical ML',
            'Experiment Tracking',
            'Distributed Computing'
        ],
        'description': [
            'Production-ready ML framework',
            'Flexible deep learning framework',
            'State-of-the-art NLP tools',
            'Traditional ML algorithms',
            'ML experiment management',
            'Scalable ML computing'
        ],
        'link': [
            'https://tensorflow.org',
            'https://pytorch.org',
            'https://huggingface.co',
            'https://scikit-learn.org',
            'https://wandb.ai',
            'https://ray.io'
        ]
    }
    return pd.DataFrame(data)

def create_resources_data():
    data = {
        'type': [
            'Book',
            'Book',
            'Video',
            'Video',
            'Community',
            'Community'
        ],
        'name': [
            'Deep Learning Book',
            'Hands-on ML',
            'CS50 AI',
            'Fast.ai Course',
            'Papers with Code',
            'AI Research Weekly'
        ],
        'author': [
            'Ian Goodfellow',
            'Aurélien Géron',
            'Harvard',
            'Jeremy Howard',
            'Meta AI',
            'Import AI'
        ],
        'description': [
            'Comprehensive deep learning theory',
            'Practical ML implementation',
            'Introduction to AI concepts',
            'Practical deep learning',
            'ML papers with implementations',
            'AI research newsletter'
        ],
        'url': [
            'https://www.deeplearningbook.org',
            'https://github.com/ageron/handson-ml2',
            'https://cs50.harvard.edu/ai',
            'https://course.fast.ai',
            'https://paperswithcode.com',
            'https://jack-clark.net'
        ]
    }
    return pd.DataFrame(data)

def create_deepseek_data():
    data = {
        'name': [
            'DeepEP', '3FS', 'DeepGEMM', 'open-infra-index', 'profile-data',
            'awesome-deepseek-integration', 'smallpond', 'FlashMLA', 'DualPipe', 'EPLB',
            'DeepSeek-VL2', 'DeepSeek-V3', 'DeepSeek-R1', 'Janus', 'DeepSeek-V2',
            'DeepSeek-Coder-V2', 'ESFT', 'DreamCraft3D', 'DeepSeek-Prover-V1.5',
            'DeepSeek-Coder', 'DeepSeek-VL', 'DeepSeek-Math', 'awesome-deepseek-coder',
            'DeepSeek-LLM', 'DeepSeek-MoE'
        ],
        'url': [f'https://github.com/deepseek-ai/{repo}' for repo in [
            'DeepEP', '3FS', 'DeepGEMM', 'open-infra-index', 'profile-data',
            'awesome-deepseek-integration', 'smallpond', 'FlashMLA', 'DualPipe', 'EPLB',
            'DeepSeek-VL2', 'DeepSeek-V3', 'DeepSeek-R1', 'Janus', 'DeepSeek-V2',
            'DeepSeek-Coder-V2', 'ESFT', 'DreamCraft3D', 'DeepSeek-Prover-V1.5',
            'DeepSeek-Coder', 'DeepSeek-VL', 'DeepSeek-Math', 'awesome-deepseek-coder',
            'DeepSeek-LLM', 'DeepSeek-MoE'
        ]],
        'description': [
            'ไลบรารีการสื่อสารแบบ Expert-Parallel ที่มีประสิทธิภาพสูง',
            'ระบบไฟล์กระจายประสิทธิภาพสูง สำหรับการฝึกและ Inference AI',
            'Kernel GEMM แบบ FP8 ที่มีประสิทธิภาพสูง',
            'เครื่องมือโครงสร้างพื้นฐาน AI สำหรับ AGI',
            'วิเคราะห์การทับซ้อนใน DeepSeek-V3/R1',
            'วิธีผสาน DeepSeek API เข้ากับซอฟต์แวร์',
            'เฟรมเวิร์กประมวลผลข้อมูลน้ำหนักเบา',
            'Kernel ถอดรหัส MLA ประสิทธิภาพสูง',
            'Pipeline Parallelism แบบสองทิศทาง',
            'Expert Parallelism Load Balancer',
            'โมเดล Vision-Language แบบ MoE',
            'โมเดล MoE ขนาด 671B Parameters',
            'โมเดล Reasoning ด้วย RL',
            'โมเดล Multimodal เข้าใจหลายรูปแบบ',
            'โมเดล MoE 236B Parameters',
            'โมเดลเขียนโค้ดอัจฉริยะ',
            'เทคนิค Expert Specialized Fine-Tuning',
            'สร้างโมเดล 3D ด้วย Diffusion',
            'โมเดลพิสูจน์คณิตศาสตร์',
            'โมเดลเขียนโค้ดอัตโนมัติ',
            'โมเดล Vision-Language เข้าใจโลกจริง',
            'โมเดลแก้ปัญหาคณิตศาสตร์',
            'รวมโปรเจกต์ DeepSeek Coder',
            'โมเดลภาษาสำหรับงานทั่วไป',
            'โมเดล MoE ประสิทธิภาพสูง'
        ],
        'type': [
            'Library', 'System', 'Kernel', 'Tool', 'Analysis',
            'Integration', 'Framework', 'Kernel', 'Algorithm', 'Tool',
            'Model', 'Model', 'Model', 'Model', 'Model',
            'Model', 'Technique', 'Model', 'Model', 'Model',
            'Model', 'Model', 'Collection', 'Model', 'Model'
        ]
    }
    return pd.DataFrame(data)

def save_all_data():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create and save all datasets
    fundamentals_df = create_fundamentals_data()
    notebooks_df = create_notebooks_data()
    courses_df = create_courses_data()
    tools_df = create_tools_data()
    resources_df = create_resources_data()
    deepseek_df = create_deepseek_data()
    
    # Save to CSV files
    fundamentals_df.to_csv('data/fundamentals.csv', index=False)
    notebooks_df.to_csv('data/notebooks.csv', index=False)
    courses_df.to_csv('data/courses.csv', index=False)
    tools_df.to_csv('data/tools.csv', index=False)
    resources_df.to_csv('data/resources.csv', index=False)
    deepseek_df.to_csv('data/deepseek.csv', index=False)

def save_all_data_as_json():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Create all datasets
    fundamentals_df = create_fundamentals_data()
    notebooks_df = create_notebooks_data()
    courses_df = create_courses_data()
    tools_df = create_tools_data()
    resources_df = create_resources_data()
    deepseek_df = create_deepseek_data()

    # Combine all data into a single dictionary
    all_data = {
        'fundamentals': fundamentals_df.to_dict(orient='records'),
        'notebooks': notebooks_df.to_dict(orient='records'),
        'courses': courses_df.to_dict(orient='records'),
        'tools': tools_df.to_dict(orient='records'),
        'resources': resources_df.to_dict(orient='records'),
        'deepseek': deepseek_df.to_dict(orient='records')
    }

    # Save to a JSON file
    import json
    with open('data/all_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
