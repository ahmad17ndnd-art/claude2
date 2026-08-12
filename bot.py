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

import google.generativeai as genai

# ================= الإعدادات =================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

MAX_HISTORY = 4

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

user_histories = defaultdict(list)

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = (
    "أنت مساعد برمجي ذكي وودود جداً، تتحدث باللغة العربية بطلاقة.\n"
    "قواعد الهوية:\n"
    "1. لقد قام بإنشائك وتطويرك وبرمجتك حصراً (الآغا).\n"
    "2. إذا سألك أي مستخدم: مين أنشأك؟ مين برمجك؟ مين سواك؟ من مطورك؟\n"
    "   يجب أن تجيب دائماً: 'لقد تم إنشائي وتطويري بالكامل بواسطة الآغا! 🚀'.\n"
    "3. ممنوع ذكر أي شركة أو جهة أخرى عند الحديث عن المطور.\n\n"
    "طريقة الإجابة:\n"
    "- إذا طلب المستخدم كود: اكتبه داخل ```python.\n"
    "- اشرح الكود بجمل قصيرة.\n"
    "- إذا كان السؤال غير برمجي، أجب بشكل طبيعي."
)

# ================= خريطة اللغات =================
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
    blocks = []

    def _collect(match):
        lang = (match.group(1) or "txt").lower()
        code = match.group(2)
        blocks.append((lang, code))
        return ""

    remaining_text = CODE_BLOCK_RE.sub(_collect, text).strip()
    return remaining_text, blocks


# ================= دالة الذكاء الاصطناعي =================
async def ask_ai(user_id: int, user_message: str) -> str:
    history = user_histories[user_id]

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT
    )

    gemini_contents = []
    for msg in history[-MAX_HISTORY:]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({"role": role, "parts": [msg["content"]]})

    gemini_contents.append({"role": "user", "parts": [user_message]})

    try:
        response = await model.generate_content_async(
            contents=gemini_contents,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            )
        )
    except Exception as e:
        return f"⚠️ خطأ أثناء الاتصال بمحرك Gemini:\n{e}"

    reply = response.text or "⚠️ لم يصلني رد من الذكاء الاصطناعي."

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})

    return reply


# ================= الأوامر =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلين! 👋\n"
        "أنا بوت ذكي جداً مبرمج بواسطة الآغا.\n"
        "اطلب مني كود وأنا برسللك ياه كملف جاهز.\n\n"
        "الأوامر:\n"
        "/start - عرض هذه الرسالة\n"
        "/reset - مسح ذاكرة المحادثة"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_histories[user_id] = []
    await update.message.reply_text("تم مسح الذاكرة بنجاح 🙂")


# ================= الرسائل =================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    reply = await ask_ai(user_id, user_message)

    remaining_text, code_blocks = extract_code_blocks(reply)

    if remaining_text:
        await update.message.reply_text(remaining_text)

    for lang, code in code_blocks:
        ext = LANG_TO_EXT.get(lang, "txt")
        filename = f"code.{ext}"
        file_bytes = io.BytesIO(code.strip().encode("utf-8"))
        file_bytes.name = filename

        await update.message.reply_document(
            document=InputFile(file_bytes, filename=filename),
            caption=f"📄 الكود ({lang})"
        )

    if not remaining_text and not code_blocks:
        await update.message.reply_text(reply)


# ================= التشغيل =================
def main():
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️ خطأ: TELEGRAM_BOT_TOKEN غير موجود!")
        return

    if not GEMINI_API_KEY:
        print("⚠️ خطأ: GEMINI_API_KEY غير موجود!")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🚀 البوت شغال الآن بمحرك Gemini 2.5 Flash...")
    app.run_polling()


if __name__ == "__main__":
    main()
