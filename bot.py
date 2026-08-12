"""
بوت تلجرام ذكي - يجاوب على الأسئلة وبيرسل الأكواد كملفات
يستخدم Groq API (مجاني) كمحرك ذكاء اصطناعي
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
from groq import Groq

# ============ الإعدادات ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "ضع_توكن_البوت_هون")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "ضع_مفتاح_Groq_هون")
MODEL_NAME = "llama-3.3-70b-versatile"  # موديل مجاني وقوي عبر Groq

# عدد الرسائل السابقة يلي بيحتفظ فيها البوت بذاكرة كل مستخدم
MAX_HISTORY = 20

# ============ تجهيز اللوغ ============
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============ عميل Groq ============
groq_client = Groq(api_key=GROQ_API_KEY)

# ذاكرة محادثة بسيطة لكل مستخدم (في الرام، بتنمسح لما يعيد تشغيل البوت)
user_histories = defaultdict(list)

SYSTEM_PROMPT = (
    "أنت مساعد برمجي ذكي وودود، بتحكي عربي وبتفهم إنجليزي كمان. "
    "لما حدا يطلب منك كود، اكتبه كامل وواضح جوا code block مع تحديد اللغة "
    "(مثلاً ```python أو ```javascript). "
    "اشرح الكود بجمل قصيرة ومفيدة قبل أو بعد الكود. "
    "إذا السؤال مش برمجي، جاوب بشكل طبيعي ومفيد."
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
    """يرسل الرسالة إلى Groq ويرجع الرد"""
    history = user_histories[user_id]
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_HISTORY:]

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=4000,
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلين! 👋\n"
        "أنا بوت ذكي بساعدك بالبرمجة وبجاوب على أي سؤال.\n"
        "اطلب مني كود وأنا برسللك ياه كملف جاهز.\n\n"
        "الأوامر:\n"
        "/start - عرض هاد الرسالة\n"
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
    if "ضع_" in TELEGRAM_BOT_TOKEN or "ضع_" in GROQ_API_KEY:
        print("⚠️  لازم تحط توكن البوت ومفتاح Groq قبل ما تشغل البوت (شوف README.md)")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🚀 البوت شغال...")
    app.run_polling()


if __name__ == "__main__":
    main()
