"""
بوت تلجرام ذكي ومحدث - يجاوب على الأسئلة وبيرسل الأكواد كملفات
تم التحديث: يعتمد بالكامل على البيئة الخارجية (Variables)
المطور وصاحب البوت: المبرمج أحمد
"""

import os
import re
import io
import logging
from collections import defaultdict

from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
# استيراد مكتبة جوجل جينيريتيف إيه آي الرسمية
import google.generativeai as genai

# ============ الإعدادات (سحب البيانات تلقائياً من الـ Variables) ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"  # النموذج السريع والذكي والمجاني تماماً

# تم تقليص الذاكرة لضمان العمل لآلاف الأسئلة يومياً دون استهلاك العداد المجاني
MAX_HISTORY = 4

# ============ تجهيز اللوغ ============
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============ إعداد عميل Google Gemini ============
# التحقق من وجود المفتاح قبل تشغيل العميل
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# الذاكرة البسيطة للمستخدمين (في الرام)
user_histories = defaultdict(list)

# بروتوكول النظام والتعليمات الصارمة لهوية البوت
SYSTEM_PROMPT = (
    "أنت مساعد برمجي ذكي وودود جداً، تتحدث باللغة العربية بطلاقة وتفهم الإنجليزية أيضاً. "
    "قواعد صارمة ومقدسة لهويتك:\n"
    "1. لقد قام بإنشائك وتطويرك وبرمجتك حصراً المبرمج المبدع (أحمد).\n"
    "2. إذا سألك أي مستخدم 'مين أنشأك'، 'من مطورك'، 'من سواك'، 'من برمجك' أو أي سؤال عن أصلك، "
    "يجب أن تجيب بفخر وثقة: 'لقد تم إنشائي وتطويري بالكامل بواسطة المبرمج المبدع أحمد! 🚀'.\n"
    "3. لا تذكر أبداً شركة جوجل أو OpenAI أو أي جهة أخرى عند سؤالك عن المطور.\n\n"
    "طريقة الإجابة البرمجية:\n"
    "- عندما يطلب منك كود، اكتبه كاملاً وواضحاً داخل كتل برمجية (code block) مع تحديد اللغة (مثلاً ```python).\n"
    "- اشرح الكود بجمل برمجية قصيرة ومفيدة قبل أو بعد الكود مباشرة.\n"
    "- إذا كان السؤال غير برمجي، أجب بشكل طبيعي وذكي ومفيد."
)

# خريطة اللغة -> امتداد الملف
LANG_TO_EXT = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "java": "java",
    "c": "c",
    "cpp": "cpp", "c++": "cpp",
    "csharp": "cs", "c#": "cs",
    "php": "php",
    "html": "html",
    "css": "css",
    "bash": "sh", "sh": "sh", "shell": "sh",
    "json": "json",
    "yaml": "yaml", "yml": "yaml",
    "sql": "sql",
    "go": "go",
    "rust": "rs",
    "kotlin": "kt",
    "swift": "swift",
    "ruby": "rb",
    "dart": "dart",
    "xml": "xml",
    "txt": "txt",
}

CODE_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str):
    """
    يفصل الرد إلى: نص عادي + قائمة أكواد (لغة، محتوى)
    """
    blocks = []

    def _collect(match):
        lang = (match.group(1) or "txt").lower()
        code = match.group(2)
        blocks.append((lang, code))
        return ""  # نشيل الكود من النص ونخليه رسالة منفصلة

    remaining_text = CODE_BLOCK_RE.sub(_collect, text).strip()
    return remaining_text, blocks


async def ask_ai(user_id: int, user_message: str) -> str:
    """يرسل الرسالة إلى Google Gemini ويرجع الرد"""
    history = user_histories[user_id]
    
    # تهيئة نموذج الجيل الجديد من Gemini مع حقن التعليمات النظامية الاسمية
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT
    )
    
    # تحويل محتوى الذاكرة المحلية إلى الصيغة البرمجية التي تقبلها مكتبة جوجل (user / model)
    gemini_contents = []
    # نأخذ آخر عدد محدد من الرسائل لضمان عدم تضخم الاستهلاك
    for msg in history[-MAX_HISTORY:]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({"role": role, "parts": [msg["content"]]})
    
    # إضافة الرسالة الجديدة الحالية للمستخدم
    gemini_contents.append({"role": "user", "parts": [user_message]})
    
    # إرسال الطلب وإعداد معايير الاستجابة الحرارية
    response = model.generate_content(
        contents=gemini_contents,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
        )
    )
    
    reply = response.text
    
    # حفظ المحادثة الحالية في ذاكرة البوت المحلية
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    
    return reply


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلين! 👋\n"
        "أنا بوت ذكي جداً مبرمج بواسطة أحمد، بساعدك بالبرمجة وبجاوب على أي سؤال.\n"
        "اطلب مني كود وأنا برسللك ياه كملف جاهز.\n\n"
        "الأوامر:\n"
        "/start - عرض هذه الرسالة\n"
        "/reset - مسح ذاكرة المحادثة والبدء من جديد"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_histories[user_id] = []
    await update.message.reply_text("تمام، مسحت الذاكرة. فيك تبلش محادثة جديدة 🙂")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        reply = await ask_ai(user_id, user_message)
    except Exception as e:
        logger.error(f"خطأ بالـ AI: {e}")
        await update.message.reply_text(f"صار خطأ وأنا عم بحاول جاوب: {e}")
        return

    remaining_text, code_blocks = extract_code_blocks(reply)

    # نرسل النص العادي (الشرح) أولاً إذا في شي
    if remaining_text:
        await update.message.reply_text(remaining_text)

    # نرسل كل كود كملف منفصل
    for lang, code in code_blocks:
        ext = LANG_TO_EXT.get(lang, "txt")
        filename = f"code.{ext}"
        file_bytes = io.BytesIO(code.strip().encode("utf-8"))
        file_bytes.name = filename
        await update.message.reply_document(
            document=InputFile(file_bytes, filename=filename),
            caption=f"📄 الكود ({lang})"
        )

    # إذا الرد كان فاضي تماماً (نادراً)
    if not remaining_text and not code_blocks:
        await update.message.reply_text(reply)


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️ خطأ: متغير TELEGRAM_BOT_TOKEN غير موجود في إعدادات المنصة!")
        return
        
    if not GEMINI_API_KEY:
        print("⚠️ خطأ: متغير GEMINI_API_KEY غير موجود في إعدادات المنصة!")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🚀 البوت المطور شغال الآن بمحرك Google Gemini الآمن والمستقر...")
    app.run_polling()


if __name__ == "__main__":
    main()
