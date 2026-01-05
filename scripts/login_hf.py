"""
Simple HuggingFace authentication script.

This script will prompt you for your HuggingFace token and save it
for future use. You only need to do this once.

Usage:
    python scripts/login_hf.py

Get your token at: https://huggingface.co/settings/tokens
(Create a token with 'Write' access)
"""

from huggingface_hub import login

print("=" * 60)
print("HUGGINGFACE AUTHENTICATION")
print("=" * 60)
print()
print("To upload datasets, you need a HuggingFace access token.")
print()
print("Steps:")
print("  1. Go to: https://huggingface.co/settings/tokens")
print("  2. Click 'New token'")
print("  3. Give it a name (e.g., 'exoskeleton-upload')")
print("  4. Select 'Write' access")
print("  5. Click 'Generate token'")
print("  6. Copy the token (starts with 'hf_...')")
print()
print("=" * 60)
print()

try:
    # This will prompt for the token
    login()

    print()
    print("=" * 60)
    print("✅ SUCCESS! You are now authenticated with HuggingFace")
    print("=" * 60)
    print()
    print("Your token has been saved. You can now upload datasets with:")
    print("  python scripts/upload_to_hf.py --repo MacExo/exoData")
    print()

except KeyboardInterrupt:
    print("\n\nAuthentication cancelled.")
    exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease make sure you:")
    print("  1. Have a valid HuggingFace account")
    print("  2. Created a token with 'Write' access")
    print("  3. Copied the full token (starts with 'hf_')")
    exit(1)
