# API Keys Setup Guide

This guide explains how to set up `OPENAI_API_KEY` and `TAVILY_API_KEY` for the News Sentinel Agent feature.

## Method 1: Streamlit Secrets (Recommended for Local Development)

The easiest way to set up API keys for local development is using Streamlit's secrets file.

### Steps:

1. **Create or edit the secrets file:**
   - Location: `.streamlit/secrets.toml`
   - This file is already in `.gitignore`, so your keys won't be committed to git

2. **Add your API keys in TOML format:**
   ```toml
   OPENAI_API_KEY = "sk-your-openai-api-key-here"
   TAVILY_API_KEY = "tvly-your-tavily-api-key-here"
   ```

3. **Restart your Streamlit app:**
   ```bash
   streamlit run app.py
   ```

### Example `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-proj-..."
TAVILY_API_KEY = "tvly-dev-..."
```

## Method 2: Environment Variables (Recommended for Production)

For production deployments or when you don't want to use Streamlit secrets, use environment variables.

### On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY = "sk-your-openai-api-key-here"
$env:TAVILY_API_KEY = "tvly-your-tavily-api-key-here"
streamlit run app.py
```

### On Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=sk-your-openai-api-key-here
set TAVILY_API_KEY=tvly-your-tavily-api-key-here
streamlit run app.py
```

### On Linux/Mac:
```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export TAVILY_API_KEY="tvly-your-tavily-api-key-here"
streamlit run app.py
```

### Permanent Setup (Windows):
1. Open System Properties → Environment Variables
2. Add new User variables:
   - `OPENAI_API_KEY` = `sk-your-key-here`
   - `TAVILY_API_KEY` = `tvly-your-key-here`

### Permanent Setup (Linux/Mac):
Add to `~/.bashrc` or ~/.zshrc`:
```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export TAVILY_API_KEY="tvly-your-tavily-api-key-here"
```

## Method 3: In-App Input (Temporary)

You can also enter the OpenAI API key directly in the app UI:
1. Go to the "News and Sentiments" tab
2. Expand "OpenAI Configuration"
3. Enter your API key in the text field
4. This only works for the current session

**Note:** Tavily API key must be set via Method 1 or 2 (not available in UI).

## Priority Order

The app checks for API keys in this order:
1. **Environment variables** (highest priority)
2. **Streamlit secrets** (`.streamlit/secrets.toml`)
3. **In-app input** (OpenAI only, for current session)

## Getting API Keys

### OpenAI API Key:
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (you won't be able to see it again!)

### Tavily API Key:
1. Go to https://tavily.com/
2. Sign up for an account
3. Navigate to your API keys section
4. Copy your API key

## Security Notes

⚠️ **Important:**
- Never commit API keys to git (`.streamlit/secrets.toml` is already in `.gitignore`)
- Don't share your API keys publicly
- Rotate keys if they're accidentally exposed
- Use environment variables in production deployments

## Troubleshooting

### "OpenAI API key not found" error:
- Check that the key is set correctly in secrets.toml or environment
- Restart the Streamlit app after adding keys
- Verify the key format (should start with `sk-` for OpenAI)

### "Tavily API key not found" error:
- Only appears if you select "Tavily" as the backend
- You can use "DuckDuckGo" backend without any API key (it's free)
- Check that the key is set correctly if using Tavily

### Keys not working:
- Verify keys are valid and have sufficient credits/quota
- Check for typos or extra spaces
- Ensure you're using the correct format (TOML format for secrets.toml)

