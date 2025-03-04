import pandas as pd
import os
from datetime import datetime

def create_section_header(title, emoji="📚"):
    return f"\n## {emoji} {title}\n\n"

def create_subsection_header(title, emoji="🔹"):
    return f"\n### {emoji} {title}\n\n"

def create_markdown_table(df, group_by=None):
    if group_by:
        tables = []
        for group_name, group_df in df.groupby(group_by):
            tables.append(f"#### {group_name}\n")
            group_df = group_df.drop(columns=[group_by])
            # Create header
            tables.append("| " + " | ".join(group_df.columns) + " |")
            tables.append("| " + " | ".join(["---" for _ in group_df.columns]) + " |")
            # Add rows
            for _, row in group_df.iterrows():
                row_values = [str(val).replace("\n", "<br>") for val in row]
                tables.append("| " + " | ".join(row_values) + " |")
            tables.append("\n")
        return "\n".join(tables)
    else:
        # Create header
        markdown = "| " + " | ".join(df.columns) + " |\n"
        markdown += "| " + " | ".join(["---" for _ in df.columns]) + " |\n"
        # Add rows
        for _, row in df.iterrows():
            row_values = [str(val).replace("\n", "<br>") for val in row]
            markdown += "| " + " | ".join(row_values) + " |\n"
        return markdown

def update_readme():
    # Read the CSV files
    fundamentals_df = pd.read_csv('data/fundamentals.csv')
    notebooks_df = pd.read_csv('data/notebooks.csv')
    courses_df = pd.read_csv('data/courses.csv')
    tools_df = pd.read_csv('data/tools.csv')
    resources_df = pd.read_csv('data/resources.csv')
    deepseek_df = pd.read_csv('data/deepseek.csv')

    # Create README content
    content = []
    
    # Title and introduction
    content.extend([
        "# AI/LLM Learning Resources 2025 🚀",
        "",
        f"*อัปเดตล่าสุด: {datetime.now().strftime('%d %B %Y')}*",
        "",
        "*คลังรวมแหล่งเรียนรู้สำหรับการพัฒนาด้าน AI และ Large Language Models (LLMs) " 
        "ครอบคลุมตั้งแต่พื้นฐานไปจนถึงการประยุกต์ใช้งานขั้นสูง*",
        "",
        "---"
    ])

    # Learning Path
    content.extend([
        create_section_header("แนะนำเส้นทางการเรียนรู้", "🎯"),
        "1. **เริ่มต้นด้วยพื้นฐาน** - คณิตศาสตร์และ Python",
        "2. **ฝึกปฏิบัติ** - ผ่าน Notebooks และ Projects",
        "3. **เรียนรู้ขั้นสูง** - คอร์สออนไลน์และการประยุกต์ใช้",
        "4. **พัฒนาต่อยอด** - ใช้เครื่องมือและทรัพยากรเพิ่มเติม",
        ""
    ])

    # Previous sections remain the same...

    # DeepSeek AI Repositories
    content.extend([
        create_section_header("DeepSeek AI Repositories", "🚀"),
        "คลังเก็บโค้ดและโมเดลจาก DeepSeek AI แบ่งตามประเภทการใช้งาน:",
        "",
        create_markdown_table(deepseek_df, group_by='type')
    ])

    # JSON Data
    content.extend([
        create_section_header("JSON Data", "📊"),
        "ข้อมูลทั้งหมดในรูปแบบ JSON:",
        "",
        "- [all_data.json](data/all_data.json)",
        ""
    ])


    # How to Proceed section
    content.extend([
        create_section_header("How to Proceed", "🚀"),
        "1. **เลือกคลัง**: เริ่มจากคลังที่ตรงกับงานของคุณ (เช่น DeepSeek-Coder สำหรับโค้ด, DeepSeek-V3 สำหรับ LLM)",
        "2. **ดาวน์โหลด**: ใช้ `git clone <URL>` หรือดาวน์โหลด ZIP จาก GitHub",
        "3. **ติดตั้ง**: รัน `pip install -r requirements.txt` หรือตามคำแนะนำใน README",
        "4. **ทดลอง**: ใช้ Script หรือ API ตามตัวอย่าง, ปรับแต่งตามความต้องการ",
        "5. **Hardware**: เตรียม GPU (แนะนำ 16GB+ VRAM) หรือใช้ Cloud เช่น Colab Pro",
        "",
        "---",
        "",
        "*Made with ❤️ by the AI Learning Community*"
    ])

    # Write to README.md
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

    print("README.md has been updated successfully.")

if __name__ == "__main__":
    update_readme()
