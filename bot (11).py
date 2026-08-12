"""
بوت تلجرام ذكي - يجاوب على الأسئلة، يرسل الأكواد كملفات، ويحلل الصور والملفات
يستخدم Groq كمحرك أساسي، وينتقل تلقائياً لـ Gemini إذا خلص الحد اليومي (fallback)
"""

import os
import re
import io
import base64
import logging
import datetime
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
from groq import RateLimitError as GroqRateLimitError

try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False

# ============ الإعدادات ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "ضع_توكن_البوت_هون")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "ضع_مفتاح_Groq_هون")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # اختياري، للـ fallback + تحليل الصور/الملفات

GROQ_MODEL = "llama-3.1-8b-instant"       # حد يومي مرتفع (~14,400/يوم)
GROQ_VISION_MODEL = "llama-4-scout-17b-16e-instruct"  # موديل Groq يلي بيشوف صور
GEMINI_MODEL = "gemini-2.5-flash"         # موديل Gemini للنص والصور والملفات

MAX_HISTORY = 20  # عدد الرسائل السابقة المحفوظة بذاكرة كل مستخدم
MAX_CONTINUATIONS = 6  # كم مرة يقدر البوت "يكمل" رد اتقطع لأنو طويل (كل مرة ~500-4000 سطر إضافية)
FORCE_GEMINI_CHAR_THRESHOLD = 20000  # ملفات أكبر من هيك بتروح مباشرة لـ Gemini (سياقه أوسع بكتير)

# امتدادات الملفات النصية/الكودية يلي منقدر نقراها كنص
TEXT_FILE_EXTENSIONS = {
    "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h", "hpp", "cs",
    "php", "html", "css", "sh", "json", "yaml", "yml", "sql", "go", "rs",
    "kt", "swift", "rb", "dart", "xml", "txt", "md", "csv", "log", "ini",
    "env", "toml",
}

# ============ تجهيز اللوغ ============
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============ إعداد العملاء (Groq + Gemini) ============
groq_client = Groq(api_key=GROQ_API_KEY)

gemini_available = bool(GEMINI_API_KEY) and GEMINI_SDK_AVAILABLE
if gemini_available:
    genai.configure(api_key=GEMINI_API_KEY)

# حالة مشتركة: هل Groq معطل مؤقتاً (خلص حده اليومي)؟ بيتصفر تلقائياً كل يوم جديد
provider_state = {"groq_disabled_date": None}


def groq_is_available() -> bool:
    return provider_state["groq_disabled_date"] != datetime.date.today()


def disable_groq_for_today():
    provider_state["groq_disabled_date"] = datetime.date.today()
    logger.warning("Groq وصل لحده اليومي، البوت رح يستخدم Gemini لباقي اليوم")


# ذاكرة محادثة موحدة لكل مستخدم (نفس الفورمات لأي مزوّد)
user_histories = defaultdict(list)

SYSTEM_PROMPT = (
    "أنت مساعد برمجي ذكي وودود، بتحكي عربي وبتفهم إنجليزي كمان. "
    "لما حدا يطلب منك كود، اكتبه كامل وواضح جوا code block مع تحديد اللغة "
    "(مثلاً ```python أو ```javascript). "
    "اشرح الكود بجمل قصيرة ومفيدة قبل أو بعد الكود. "
    "إذا حدا بعتلك كود أو صورة كود أو ملف وطلب تحليل، دور على الأخطاء البرمجية "
    "والمشاكل المحتملة واشرحها بوضوح واقترح الإصلاح. "
    "إذا السؤال مش برمجي، جاوب بشكل طبيعي ومفيد."
)

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
    """يفصل الرد إلى: نص عادي + قائمة أكواد (لغة، محتوى)"""
    blocks = []

    def _collect(match):
        lang = (match.group(1) or "txt").lower()
        code = match.group(2)
        blocks.append((lang, code))
        return ""

    remaining_text = CODE_BLOCK_RE.sub(_collect, text).strip()
    return remaining_text, blocks


# ============ طبقة استدعاء الذكاء الاصطناعي (نص) ============

def _call_groq_text(messages):
    """يرجع (النص، هل_انقطع). الانقطاع بيصير إذا الرد وصل للحد الأقصى المسموح بطلب واحد."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=8000,  # هاد أقصى حد ممكن لهاد الموديل تحديداً على Groq (سقف ثابت من المزوّد)
    )
    choice = response.choices[0]
    truncated = choice.finish_reason == "length"
    return choice.message.content, truncated


def _call_gemini_text(messages):
    """يرجع (النص، هل_انقطع). Gemini بيسمح بردود أطول بكتير من Groq بالطلب الواحد."""
    system_text = ""
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        elif m["role"] == "user":
            contents.append({"role": "user", "parts": [m["content"]]})
        elif m["role"] == "assistant":
            contents.append({"role": "model", "parts": [m["content"]]})

    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_text.strip() or None)
    generation_config = {"max_output_tokens": 65536}
    try:
        # نطفي "التفكير الداخلي" عشان كل الحصة تروح للرد الفعلي مش للتفكير
        response = model.generate_content(
            contents,
            generation_config={**generation_config, "thinking_config": {"thinking_budget": 0}},
        )
    except Exception:
        response = model.generate_content(contents, generation_config=generation_config)

    candidate = response.candidates[0]
    finish_reason = str(candidate.finish_reason)
    truncated = "MAX_TOKENS" in finish_reason
    return response.text, truncated


def _call_provider(provider: str, messages):
    """يستدعي مزوّد محدد بالاسم، يرجع (نص، انقطع)"""
    if provider == "groq":
        return _call_groq_text(messages)
    return _call_gemini_text(messages)


async def ask_ai(user_id: int, user_message: str, force_provider: str = None) -> str:
    """
    يرسل الرسالة لأول مزوّد متاح (Groq)، وإذا خلص حده اليومي يتحول تلقائياً
    لـ Gemini بدون ما تنقطع ذاكرة المحادثة. إذا الرد طويل وانقطع، بيطلب من
    نفس المزوّد يكمل تلقائياً (لحد MAX_CONTINUATIONS مرة) ويلزق الأجزاء مع بعض.
    """
    history = user_histories[user_id]
    history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_HISTORY:]

    provider_used = force_provider
    full_reply = None

    if provider_used is None:
        if groq_is_available():
            try:
                full_reply, truncated = _call_groq_text(messages)
                provider_used = "groq"
            except GroqRateLimitError:
                disable_groq_for_today()
            except Exception as e:
                logger.error(f"خطأ Groq: {e}")

        if full_reply is None:
            if not gemini_available:
                raise RuntimeError(
                    "خلص الحد اليومي لـ Groq وما في مفتاح Gemini احتياطي معرّف "
                    "(GEMINI_API_KEY). ضيفو عشان البوت يضل شغال."
                )
            full_reply, truncated = _call_gemini_text(messages)
            provider_used = "gemini"
    else:
        full_reply, truncated = _call_provider(provider_used, messages)

    # ============ استكمال تلقائي إذا الرد اتقطع (طويل جداً لطلب واحد) ============
    continuation_messages = list(messages) + [{"role": "assistant", "content": full_reply}]
    rounds = 0
    while truncated and rounds < MAX_CONTINUATIONS:
        rounds += 1
        continuation_messages.append({
            "role": "user",
            "content": "كمل تماماً من حيث وقفت، بدون ما تعيد أي شي كتبته قبل ولا تكرر مقدمات.",
        })
        try:
            extra_text, truncated = _call_provider(provider_used, continuation_messages)
        except Exception as e:
            logger.error(f"خطأ بالاستكمال التلقائي: {e}")
            break
        full_reply += extra_text
        continuation_messages.append({"role": "assistant", "content": extra_text})

    history.append({"role": "assistant", "content": full_reply})
    return full_reply


# ============ طبقة تحليل الصور والملفات ============

async def analyze_image(image_bytes: bytes, mime_type: str, prompt: str) -> str:
    """يحلل صورة (سكرين شوت كود، رسم بياني...) عبر Gemini أولاً وGroq احتياطياً"""
    if gemini_available:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
            response = model.generate_content(
                [prompt, {"mime_type": mime_type, "data": image_bytes}]
            )
            return response.text
        except Exception as e:
            logger.error(f"خطأ Gemini بتحليل الصورة: {e}")

    if groq_is_available():
        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            response = groq_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                        ],
                    },
                ],
                temperature=0.5,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except GroqRateLimitError:
            disable_groq_for_today()
        except Exception as e:
            logger.error(f"خطأ Groq بتحليل الصورة: {e}")

    raise RuntimeError("ما قدرت حلل الصورة، جرب كمان شوي أو تأكد من مفاتيح الـ API.")


async def analyze_pdf(file_bytes: bytes, prompt: str) -> str:
    """يحلل ملف PDF عبر Gemini فقط (Groq ما بيدعم PDF مباشرة)"""
    if not gemini_available:
        raise RuntimeError("تحليل ملفات PDF بيحتاج مفتاح Gemini (GEMINI_API_KEY) مش معرّف حالياً.")

    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(
        [prompt, {"mime_type": "application/pdf", "data": file_bytes}]
    )
    return response.text


# ============ أوامر ورسائل تلجرام ============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلين! 👋\n"
        "أنا بوت ذكي بساعدك بالبرمجة وبجاوب على أي سؤال.\n"
        "• اطلب مني كود وأنا برسللك ياه كملف جاهز.\n"
        "• ابعتلي صورة (سكرين شوت كود مثلاً) وبحللها وبلاقي الأخطاء.\n"
        "• ابعتلي ملف كود أو PDF وبحللو وبقلك شو فيه.\n\n"
        "الأوامر:\n"
        "/start - عرض هاد الرسالة\n"
        "/reset - مسح ذاكرة المحادثة والبدء من جديد"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_histories[user_id] = []
    await update.message.reply_text("تمام، مسحت الذاكرة. فيك تبلش محادثة جديدة 🙂")


TELEGRAM_MAX_MESSAGE_LEN = 4000  # تلجرام بيسمح لحد 4096، منترك هامش أمان


async def _reply_text_chunked(update: Update, text: str):
    """يقسم النص الطويل لأكتر من رسالة عشان ما يتجاوز حد تلجرام (4096 حرف)"""
    for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LEN):
        await update.message.reply_text(text[i:i + TELEGRAM_MAX_MESSAGE_LEN])


async def send_ai_reply(update: Update, reply: str):
    """يفصل رد الذكاء الاصطناعي لنص + أكواد، وبيرسل كل كود كملف منفصل"""
    remaining_text, code_blocks = extract_code_blocks(reply)

    if remaining_text:
        await _reply_text_chunked(update, remaining_text)

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
        await _reply_text_chunked(update, reply)


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

    await send_ai_reply(update, reply)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """لما المستخدم يبعت صورة (سكرين شوت كود مثلاً)"""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    photo = update.message.photo[-1]  # أعلى جودة
    tg_file = await photo.get_file()
    image_bytes = bytes(await tg_file.download_as_bytearray())

    caption = update.message.caption or "حلل هاد الكود/الصورة ولاقي أي أخطاء أو مشاكل فيها واشرحلي."

    try:
        reply = await analyze_image(image_bytes, "image/jpeg", caption)
    except Exception as e:
        logger.error(f"خطأ بتحليل الصورة: {e}")
        await update.message.reply_text(f"ما قدرت حلل الصورة: {e}")
        return

    user_histories[update.effective_user.id].append(
        {"role": "assistant", "content": reply}
    )
    await send_ai_reply(update, reply)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """لما المستخدم يبعت ملف (كود، نص، PDF، أو صورة كملف)"""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    document = update.message.document
    filename = document.file_name or "file"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime_type = document.mime_type or ""

    tg_file = await document.get_file()
    file_bytes = bytes(await tg_file.download_as_bytearray())

    caption = update.message.caption or "حلل هاد الملف، لاقي الأخطاء البرمجية إذا في، واشرحلي شو فيه."

    try:
        if mime_type.startswith("image/"):
            reply = await analyze_image(file_bytes, mime_type, caption)

        elif ext == "pdf" or mime_type == "application/pdf":
            reply = await analyze_pdf(file_bytes, caption)

        elif ext in TEXT_FILE_EXTENSIONS or mime_type.startswith("text/"):
            try:
                file_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                file_text = file_bytes.decode("utf-8", errors="ignore")

            if len(file_text) > 300000:
                file_text = file_text[:300000] + "\n... (تم اقتطاع باقي الملف لأنو طويل جداً)"

            prompt = f"{caption}\n\nاسم الملف: {filename}\nمحتوى الملف:\n```{ext}\n{file_text}\n```"

            # الملفات الكبيرة (آلاف الأسطر) بتروح مباشرة لـ Gemini (سياقه وحده الأقصى أوسع بكتير من Groq)
            force_provider = "gemini" if (len(file_text) > FORCE_GEMINI_CHAR_THRESHOLD and gemini_available) else None
            reply = await ask_ai(update.effective_user.id, prompt, force_provider=force_provider)

        else:
            await update.message.reply_text(
                f"نوع الملف ده ({ext or mime_type}) مش مدعوم حالياً للتحليل. "
                "بقدر أحلل: صور، PDF، وملفات كود/نص (py, js, html, json...)."
            )
            return

    except Exception as e:
        logger.error(f"خطأ بتحليل الملف: {e}")
        await update.message.reply_text(f"ما قدرت حلل الملف: {e}")
        return

    await send_ai_reply(update, reply)


def main():
    if "ضع_" in TELEGRAM_BOT_TOKEN or "ضع_" in GROQ_API_KEY:
        print("⚠️  لازم تحط توكن البوت ومفتاح Groq قبل ما تشغل البوت (شوف README.md)")
        return

    if not gemini_available:
        print("ℹ️  ما في مفتاح Gemini (GEMINI_API_KEY) — البوت رح يشتغل بـ Groq بس، "
              "بدون fallback ولا تحليل صور/PDF.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    print("🚀 البوت شغال...")
    app.run_polling()


if __name__ == "__main__":
    main()
