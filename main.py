import os
import sys
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from google.colab import userdata


# Ensure the project root is in sys.path so absolute imports work when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables from config/.env (user should copy .env.example to .env)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

# Import handler functions (to be implemented)
from bot.handlers import (
    start_handler,
    idea_handler,
    gen_handler,
    random_handler,
    setting_handler,
    lora_handler,
    loras_handler,
    clear_loras_handler,
    cancel_handler,
    model_upload_handler,
    civitai_file_choice,
)

async def main() -> None:
    """Entry point for the Telegram bot."""
    token = os.getenv("TELEGRAM_TOKEN") or userdata.get('TELEGRAM_TOKEN')
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set in environment variables.")

    # Build the application
    app = ApplicationBuilder().token(token).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("idea", idea_handler))
    app.add_handler(CommandHandler("gen", gen_handler))
    app.add_handler(CommandHandler("random", random_handler))
    app.add_handler(CommandHandler("setting", setting_handler))
    app.add_handler(CommandHandler("lora", lora_handler))
    app.add_handler(CommandHandler("loras", loras_handler))
    app.add_handler(CommandHandler("clear_loras", clear_loras_handler))
    app.add_handler(CommandHandler("cancel", cancel_handler))
    # Handler for inline button callbacks from /model_upload
    app.add_handler(CallbackQueryHandler(civitai_file_choice, pattern="^civitai_file_"))

    # Start the bot (polling mode works in Colab)
    await app.start()
    print("Bot started. Listening for updates...")
    await app.updater.start_polling()
    # Run until stopped (Ctrl+C in the notebook)
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped.")
