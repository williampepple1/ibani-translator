# Quick Vercel Setup Guide

## What Changed?

Your Ibani Translator now supports loading models from HuggingFace Hub, making it perfect for Vercel deployment without uploading large model files!

## How It Works

```
┌─────────────────┐
│  API Request    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Check Local Model (./ibani_model)  │
│  ✓ Found? → Use local model         │
│  ✗ Not found? → Continue            │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Load from HuggingFace Hub          │
│  (williampepple1/ibani-translator)  │
│  ✓ Success? → Use HF model          │
│  ✗ Failed? → Use base model         │
└─────────────────────────────────────┘
```

## Files Added/Modified

### New Files
- ✅ `api/index.py` - Vercel serverless function entry point
- ✅ `vercel.json` - Vercel configuration
- ✅ `DEPLOYMENT.md` - Comprehensive deployment guide
- ✅ `VERCEL_SETUP.md` - This quick guide
- ✅ `test_model_loading.py` - Test script for model loading

### Modified Files
- ✅ `huggingface_translator.py` - Added HuggingFace Hub support
- ✅ `api_server.py` - Added environment variable support
- ✅ `README.md` - Updated with deployment info
- ✅ `.gitignore` - Added Vercel and environment files

## Quick Deploy to Vercel

### Method 1: Vercel Dashboard (Easiest)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Vercel deployment support"
   git push origin main
   ```

2. **Deploy on Vercel**
   - Go to https://vercel.com/new
   - Import your GitHub repository
   - Add environment variable:
     - Name: `HF_MODEL_REPO`
     - Value: `williampepple1/ibani-translator`
   - Click "Deploy"

3. **Done!** 🎉
   Your API will be live at `https://your-project.vercel.app`

### Method 2: Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel

# Set environment variable
vercel env add HF_MODEL_REPO
# Enter: williampepple1/ibani-translator

# Deploy to production
vercel --prod
```

## Testing Your Deployment

```bash
# Replace YOUR_URL with your Vercel deployment URL
export API_URL="https://your-project.vercel.app"

# Test root endpoint
curl $API_URL/

# Test health check
curl $API_URL/health

# Test translation
curl -X POST "$API_URL/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am eating fish"}'
```

## Environment Variables

| Variable | Value | Required | Description |
|----------|-------|----------|-------------|
| `HF_MODEL_REPO` | `williampepple1/ibani-translator` | Yes | HuggingFace model repository |
| `LOCAL_MODEL_PATH` | `./ibani_model` | No | Local model path (for local dev) |

## Local Development

Your local setup remains unchanged:

```bash
# Install dependencies
pip install -r requirements.txt

# Run server (uses local model if available)
python api_server.py

# Or test model loading
python test_model_loading.py
```

## Benefits

✅ **No Large Files**: Don't upload 300MB+ model files to Vercel  
✅ **Automatic Updates**: Update model on HuggingFace, Vercel uses latest  
✅ **Fast Deployment**: Only code files need to be deployed  
✅ **Flexible**: Works locally with local model, remotely with HF model  
✅ **Free Tier Friendly**: Fits within Vercel's free tier limits  

## Rate Limits

Your API includes built-in protection:
- `/translate`: 20 requests/minute per IP
- `/batch-translate`: 5 requests/minute per IP (max 50 texts)
- `/health`: 30 requests/minute per IP

## What Happens on First Request?

1. Vercel function starts (cold start ~5-10 seconds)
2. Model downloads from HuggingFace Hub (~30 seconds first time)
3. Model is cached for subsequent requests
4. Future requests are fast (~100-200ms)

**Note**: First request may take 30-45 seconds. After that, it's fast!

## Troubleshooting

### "Model not loaded" error
- Check `HF_MODEL_REPO` environment variable is set
- Verify model is public on HuggingFace Hub
- Check Vercel logs: `vercel logs`

### Timeout errors
- First request takes longer (model download)
- Consider Vercel Pro for longer timeouts
- Model is cached after first download

### Rate limit errors
- Normal behavior for too many requests
- Increase limits in `api_server.py` if needed
- Returns HTTP 429 status code

## Next Steps

1. ✅ Deploy to Vercel
2. 🔧 Test all endpoints
3. 📱 Add custom domain (optional)
4. 📊 Monitor usage in Vercel dashboard
5. 🚀 Share your API!

## Support

- Full guide: See `DEPLOYMENT.md`
- API docs: Visit `/docs` on your deployment
- Issues: Check Vercel logs

---

**Your API is now ready for the world! 🌍🚀**

