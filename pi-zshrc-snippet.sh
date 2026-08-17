# Mogwai -- Raspberry Pi .zshrc addition
#
# This is NOT meant to run on the Mac. Copy the block below into the Pi's
# ~/.zshrc (e.g. `nano ~/.zshrc`, paste at the end, save) so every new
# terminal on the Pi has ANTHROPIC_API_KEY set automatically.
#
# This file does NOT contain the actual key -- only the line that loads it
# from ~/.mogwai_api_key.sh. You still need to put a real key in that file
# on the Pi itself:
#
#   cat > ~/.mogwai_api_key.sh << 'EOF'
#   export ANTHROPIC_API_KEY=sk-ant-...-your-key-here
#   EOF
#   chmod 600 ~/.mogwai_api_key.sh
#
# Worth generating a SEPARATE key for the Pi at console.anthropic.com/settings/keys
# rather than reusing the Mac's -- same account/billing either way, but it
# keeps usage and revocation independent per device.

# ---- paste from here down into the Pi's ~/.zshrc ----

if [ -f "$HOME/.mogwai_api_key.sh" ]; then
    source "$HOME/.mogwai_api_key.sh"
fi

# ---- end of block ----
