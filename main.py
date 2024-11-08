import os
import openai
from google.cloud import vision
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
import time  # مكتبة للتوقيت الوهمي

# إعداد مفاتيح API (من الأفضل حفظها في بيئة آمنة)
from dotenv import load_dotenv

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

# استخدام المتغيرات البيئية
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
openai.api_key = os.getenv("OPENAI_API_KEY")

def detect_text_from_image(image_content):
    """يكتشف النص من محتوى الصورة ويعيد النصوص المستخرجة."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if response.error.message:
        raise Exception(
            f"{response.error.message}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors"
        )
    return "\n".join([text.description for text in texts])

def analyze_text_with_gpt(page_text, page_number):
    """تحليل النص المستخرج من صفحة باستخدام ChatGPT وإرجاع الرد."""
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "  خليها كفتورة مختصرة بشكل مرتب ودقيق واذا كان فيه بيانات على شكل جدول ارسمها على شكل جدول"},
            {"role": "user", "content": f"الصفحة {page_number}:\n{page_text}"},
        ],
    )
    return gpt_response['choices'][0]['message']['content']

def process_page_sequentially(doc_path, num_pages):
    """معالجة الصفحات واحدة تلو الأخرى."""
    doc = fitz.open(doc_path)
    return_dict = {}
    
    for page_num in range(num_pages):
        page = doc[page_num]
        pix = page.get_pixmap()
        
        # تحويل الصورة إلى صيغة PIL لمعالجة Google Vision API
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(f"temp_image_{page_num}.jpg")
        
        # قراءة الصورة المؤقتة وتحليل النص
        with open(f"temp_image_{page_num}.jpg", "rb") as img_file:
            image_content = img_file.read()
            page_text = detect_text_from_image(image_content)
        
        # إرسال النص المستخرج من الصفحة إلى ChatGPT وتخزين الرد
        page_response = analyze_text_with_gpt(page_text, page_num + 1)
        return_dict[page_num] = page_response
        
        # حذف الصورة المؤقتة
        os.remove(f"temp_image_{page_num}.jpg")
        
    doc.close()
    return return_dict

# بدء واجهة Streamlit
st.set_page_config(page_title="تحليل النصوص في ملف PDF", page_icon="📄")
st.title("📄 تحليل النصوص في ملف PDF")
st.write("أداة لاستخراج وتحليل النصوص من ملفات PDF باستخدام Google Vision API و OpenAI GPT.")

uploaded_pdf = st.file_uploader("اختر ملف PDF", type="pdf")

if uploaded_pdf is not None:
    # حفظ الملف المرفوع في مسار مؤقت
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    # فتح ملف PDF باستخدام PyMuPDF لمعرفة عدد الصفحات
    doc = fitz.open(temp_pdf_path)
    num_pages = len(doc)
    doc.close()

    st.info(f"الملف يحتوي على {num_pages} صفحة. سيتم استخراج النص وتحليله، يرجى الانتظار.")

    # شريط التقدم الوهمي
    progress_bar = st.progress(0)
    
    # تحديث التقدم الوهمي بشكل تدريجي
    for i in range(100):
        progress_bar.progress(i / 100)
        time.sleep(0.20)  # الانتظار لفترة قصيرة بين كل تحديث وهمي

    # معالجة الصفحات
    return_dict = process_page_sequentially(temp_pdf_path, num_pages)

    # تحديث شريط التقدم إلى 100% بعد إكمال المعالجة
    progress_bar.progress(1.0)
    
    # حذف ملف PDF المؤقت
    os.remove(temp_pdf_path)

    # عرض الردود لكل صفحة في مربعات قابلة للطي
    st.subheader("نتائج التحليل لكل صفحة:")
    for page_num in sorted(return_dict.keys()):
        with st.expander(f"الصفحة {page_num + 1}"):
            st.write(return_dict[page_num])
else:
    st.write("يرجى تحميل ملف PDF لتحليل النصوص.")
