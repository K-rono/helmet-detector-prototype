# Deployment Guide

This guide provides detailed instructions for deploying the Helmet Detection System to various cloud platforms.

## Streamlit Cloud Deployment

### Method 1: Standard Deployment

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)** and sign in with GitHub
3. **Click "New app"** and select your forked repository
4. **Configure the deployment**:
   - **Main file path**: `app/main.py`
   - **Requirements file**: `requirements-deploy.txt`
   - **Python version**: 3.11
5. **Deploy** and wait for the build to complete

### Method 2: Using Setup Script

For more control over the deployment process:

1. **Configure the deployment**:
   - **Main file path**: `app/main.py`
   - **Requirements file**: Leave empty
   - **Setup script**: `setup.sh`
   - **Python version**: 3.11
2. **Deploy** and wait for the build to complete

### Method 3: Advanced Setup Script

For maximum control and detailed logging:

1. **Configure the deployment**:
   - **Main file path**: `app/main.py`
   - **Requirements file**: Leave empty
   - **Setup script**: `setup-advanced.sh`
   - **Python version**: 3.11
2. **Deploy** and wait for the build to complete

## Heroku Deployment

1. **Install Heroku CLI** and login
2. **Create a new Heroku app**:
   ```bash
   heroku create your-app-name
   ```
3. **Set buildpacks**:
   ```bash
   heroku buildpacks:add heroku/python
   ```
4. **Create Procfile**:
   ```
   web: streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
   ```
5. **Deploy**:
   ```bash
   git push heroku main
   ```

## Railway Deployment

1. **Connect your GitHub repository** to Railway
2. **Configure the deployment**:
   - **Build command**: `pip install -r requirements-deploy.txt`
   - **Start command**: `streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0`
3. **Deploy** and wait for the build to complete

## Google Cloud Run Deployment

1. **Create a Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY requirements-deploy.txt .
   RUN pip install -r requirements-deploy.txt
   
   COPY . .
   
   EXPOSE 8501
   CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. **Build and deploy** using Google Cloud Build

## Important Notes

### Available Detectors in Deployment

- ✅ **RCNN (Faster R-CNN)**: Fully functional
- ✅ **SSD (SSD300-VGG16)**: Fully functional
- ✅ **DETR**: Fully functional
- ❌ **YOLO Only**: Not available (requires ultralytics)
- ❌ **Hybrid (YOLO+VGG)**: Not available (requires ultralytics)

### Model Management

- **Local Models**: Place in `models/` directory for local development
- **Runtime Upload**: Users can upload models through the web interface
- **Temporary Storage**: Uploaded models are stored in session memory
- **Automatic Cleanup**: Temporary files are cleaned up when sessions end

### Environment Variables

The following environment variables are automatically set by the setup scripts:

- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_SERVER_PORT=8501`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
- `PYTHONPATH` (includes project root)

### Troubleshooting

#### Common Issues

1. **OpenCV Import Error**: 
   - **Solution**: Use `requirements-deploy.txt` which excludes OpenCV
   - **Cause**: OpenCV requires system-level graphics libraries

2. **Ultralytics Import Error**:
   - **Solution**: Use `requirements-deploy.txt` which excludes ultralytics
   - **Cause**: Ultralytics requires OpenCV internally

3. **Model Not Found**:
   - **Solution**: Upload models through the web interface
   - **Alternative**: Place models in `models/` directory

4. **Memory Issues**:
   - **Solution**: Use smaller models or increase deployment memory
   - **Note**: DETR models are typically larger than RCNN/SSD

#### Health Check

The advanced setup script includes a health check that verifies:
- Core dependencies are installed
- Available detectors are working
- YOLO detector is correctly unavailable
- Models directory exists

### Performance Optimization

1. **Use SSD detector** for fastest inference
2. **Use RCNN detector** for balanced performance
3. **Use DETR detector** for highest accuracy
4. **Enable GPU** if available (automatic detection)

### Security Considerations

1. **Model Upload**: Models are stored in temporary session memory
2. **File Size Limits**: 500MB maximum upload size
3. **Session Cleanup**: Automatic cleanup when sessions end
4. **No Persistent Storage**: Models are not permanently stored on the server
