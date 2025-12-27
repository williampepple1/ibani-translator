# Deployment Guide

This guide explains how to deploy the Ibani Translator API to Vercel using the model hosted on HuggingFace Hub.

## Architecture

The application is designed to work in two modes:

1. **Local Development**: Loads model from `./ibani_model` directory
2. **Production (Vercel)**: Automatically loads model from HuggingFace Hub if local model is not found

## Prerequisites

1. **HuggingFace Account**: Create an account at https://huggingface.co
2. **Vercel Account**: Create an account at https://vercel.com
3. **Model on HuggingFace**: Your model should be uploaded to HuggingFace Hub (e.g., `williampepple1/ibani-translator`)

## Local Setup

### 1. Environment Variables

The application supports the following environment variables:

```bash
# HuggingFace Model Repository (used when local model is not found)
HF_MODEL_REPO=williampepple1/ibani-translator

# Local Model Path (for local development)
LOCAL_MODEL_PATH=./ibani_model
```

### 2. Running Locally

#### With Local Model:
```bash
python api_server.py
```

#### With HuggingFace Model:
```bash
# Remove or rename local model directory to test HuggingFace fallback
mv ibani_model ibani_model.backup

# Set environment variable and run
export HF_MODEL_REPO=williampepple1/ibani-translator
python api_server.py
```

The server will automatically:
1. Try to load from `./ibani_model`
2. If not found, load from HuggingFace Hub
3. If HuggingFace fails, fall back to base model

## Vercel Deployment

### Step 1: Prepare Your Repository

Make sure your repository has these files:
- `api_server.py` - Main API server
- `huggingface_translator.py` - Translation logic
- `requirements.txt` - Python dependencies
- `vercel.json` - Vercel configuration
- `api/index.py` - Vercel serverless function entry point

### Step 2: Connect to Vercel

#### Option A: Using Vercel CLI

1. **Install Vercel CLI:**
```bash
npm install -g vercel
```

2. **Login to Vercel:**
```bash
vercel login
```

3. **Deploy:**
```bash
vercel
```

4. **Set Environment Variables:**
```bash
vercel env add HF_MODEL_REPO
# Enter: williampepple1/ibani-translator

vercel env add LOCAL_MODEL_PATH
# Enter: ./ibani_model
```

5. **Deploy to Production:**
```bash
vercel --prod
```

#### Option B: Using Vercel Dashboard

1. Go to https://vercel.com/dashboard
2. Click "Add New" → "Project"
3. Import your Git repository (GitHub, GitLab, or Bitbucket)
4. Configure the project:
   - **Framework Preset**: Other
   - **Root Directory**: ./
   - **Build Command**: (leave empty)
   - **Output Directory**: (leave empty)

5. **Add Environment Variables:**
   - Click "Environment Variables"
   - Add:
     - Name: `HF_MODEL_REPO`
     - Value: `williampepple1/ibani-translator`
     - Name: `LOCAL_MODEL_PATH`
     - Value: `./ibani_model`

6. Click "Deploy"

### Step 3: Test Your Deployment

Once deployed, test your API:

```bash
# Get your Vercel URL (e.g., your-project.vercel.app)
curl https://your-project.vercel.app/

# Test translation
curl -X POST "https://your-project.vercel.app/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am eating fish"}'

# Check health
curl https://your-project.vercel.app/health
```

## Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HF_MODEL_REPO` | HuggingFace Hub repository ID | `williampepple1/ibani-translator` | No |
| `LOCAL_MODEL_PATH` | Path to local model directory | `./ibani_model` | No |
| `HF_TOKEN` | HuggingFace token (for private models) | None | No |

## Troubleshooting

### Model Loading Issues

**Problem**: Model not loading from HuggingFace Hub

**Solutions**:
1. Verify your model is public on HuggingFace Hub
2. Check the repository name is correct
3. If using a private model, add `HF_TOKEN` environment variable

### Memory Issues on Vercel

**Problem**: Lambda function runs out of memory

**Solutions**:
1. Use Vercel Pro plan for more memory
2. Consider using a smaller base model
3. Optimize model with quantization


## Custom Domain

1. Go to your Vercel project settings
2. Click "Domains"
3. Add your custom domain
4. Update DNS records as instructed

## Monitoring

Vercel provides built-in monitoring:
- **Logs**: View function logs in Vercel dashboard
- **Analytics**: Track API usage and performance
- **Errors**: Automatic error tracking

## Costs

- **Vercel Hobby**: Free tier includes:
  - 100 GB-hours of serverless function execution
  - Unlimited deployments
  - Automatic HTTPS

- **Model Hosting on HuggingFace**: Free for public models

## Security

1. **HTTPS**: Automatically enabled by Vercel
2. **CORS**: Configure in `api_server.py` if needed:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Alternative Deployment Options

### Docker Deployment

See `README.md` for Docker deployment instructions.

### Other Platforms

The application can also be deployed to:
- **Railway**: Similar to Vercel, good for Python apps
- **Render**: Supports persistent storage
- **Heroku**: Traditional platform-as-a-service
- **AWS Lambda**: Using Serverless Framework
- **Google Cloud Run**: Containerized deployment

For these platforms, ensure:
1. Set environment variables for `HF_MODEL_REPO`
2. Install dependencies from `requirements.txt`
3. Expose port 8080 or configure as needed

## Support

For issues or questions:
1. Check Vercel logs: `vercel logs <deployment-url>`
2. Review HuggingFace model page for model issues
3. Test locally first to isolate the problem

---

**Happy Deploying! 🚀**

