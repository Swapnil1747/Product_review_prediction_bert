from huggingface_hub import whoami

try:
    info = whoami()
    print("✅ Logged in as:", info["name"])
except Exception as e:
    print("❌ Not logged in or token invalid:", str(e))
