# Atmospheric Cloud Classification for Gemini North Telescope

A machine learning system that automatically classifies cloud formations to assist telescope operators in making weather-related operational decisions at the Gemini North Observatory on Maunakea, Hawaii.

## 🔭 Project Overview

The Gemini North Telescope, located at 14,000 feet on Maunakea's summit, requires precise weather monitoring to protect its equipment and optimize observation time. This project enhances operators' ability to identify cloud-induced weather risks by implementing a real-time CNN-based classification system.

**Key Achievement:** Developed a transfer-learned ResNet50 model achieving **85% accuracy** on cloud pattern classification, deployed as a real-time inference system integrated with existing cloud camera infrastructure.

## 🎯 Problem Statement

Telescope operators at the Gemini Hilo Base Facility currently rely solely on manual interpretation of still images from cloud cameras to assess weather threats. This project automates cloud classification to provide operators with:
- Real-time cloud category predictions every 20 seconds
- Enhanced weather risk assessment capabilities
- Reduced telescope downtime from unexpected cloud cover

## 🧠 Technical Approach

### Model Architecture
- **Base Model:** Pre-trained ResNet50 (transfer learning)
- **Classification Categories:** 4 cloud types
  - Clear skies
  - Cirrus clouds  
  - Cumulus clouds
  - Fog/Virga
- **Training Strategy:** Gradual unfreezing with decreasing learning rates
- **Performance:** 85% test accuracy on real telescope camera data

### Dataset
- **Size:** 200+ manually labeled images from Gemini cloud cameras
- **Validation:** Cross-checked with telescope operators
- **Augmentation:** Rotation, shifts, flips, zoom, brightness variations
- **Split:** 80% training, 20% validation

### Deployment Architecture
- **Containerized Solution:** Docker deployment for scalability
- **Real-time Processing:** Processes images every 20 seconds
- **Web Interface:** Flask-based dashboard for operators
- **Infrastructure:** VM-based deployment with CPU optimization

## 🚀 Features

- **Real-time Classification:** Automated analysis of cloud camera images
- **Web Dashboard:** User-friendly interface displaying current conditions
- **Containerized Deployment:** Docker-based system for easy deployment
- **Performance Optimized:** Multi-CPU processing for efficient inference
- **Integration Ready:** Designed to integrate with existing Gemini systems

## 🛠️ Technologies Used

- **Machine Learning:** TensorFlow/Keras, ResNet50, Transfer Learning
- **Backend:** Python, Flask
- **Deployment:** Docker, GitLab CI/CD
- **Infrastructure:** Linux (Rocky Linux), VM deployment
- **Data Processing:** NumPy, OpenCV, Pandas

## 📊 Model Performance

- **Final Accuracy:** 85% on test dataset
- **Inference Speed:** Real-time processing (< 1 second per image)
- **Model Size:** Optimized for production deployment
- **Robustness:** Trained with extensive data augmentation

## 🔧 Installation & Setup

### Prerequisites
- Docker installed
- Python 3.9+
- Access to GitLab repository

### Quick Start
```bash
# Clone the repository
git clone https://gitlab.com/frances-uy/cloud_classification.git
cd cloud_classification

# Pull and run the Docker container
docker pull registry.gitlab.com/frances-uy/cloud_classification:v3
docker run -d -p 8080:8080 --name cloud_classification registry.gitlab.com/frances-uy/cloud_classification:v3

# Access the web interface
# Navigate to http://localhost:8080
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python web_app.py
```

## 📈 Results & Impact

- **Operational Benefit:** Provides telescope operators with automated cloud risk assessment
- **Accuracy Improvement:** 85% classification accuracy on real observatory data
- **Response Time:** Real-time analysis reduces manual observation workload
- **Scalability:** Containerized solution ready for production deployment

## 🔮 Future Enhancements

- Expand to additional cloud camera locations
- Implement daytime image classification
- Develop more robust alerting system for control room integration
- Continuous model improvement with operational data

## 🏗️ System Requirements

- **Minimum:** 4GB RAM, 2 CPU cores
- **Recommended:** 8GB RAM, 4+ CPU cores
- **Storage:** 2GB for model and dependencies
- **OS:** Linux (tested on Rocky Linux 8.10/9.4)

## 📄 Project Structure

```
cloud_classification/
├── cloud_classification_model.keras          # Trained model
├── web_app.py                                # Flask web application
├── real_time_classification_improved.py      # Inference engine
├── Dockerfile                               # Container configuration
├── requirements.txt                         # Python dependencies
├── templates/                              # Web interface templates
└── static/                                # Static assets
```

## 👥 Acknowledgments

Developed during internship at Gemini Observatory/NOIRLab, with guidance from telescope operators and engineering staff. Special thanks to the Gemini team for providing real operational data and domain expertise.

---

**Note:** This project demonstrates practical application of computer vision and machine learning in astronomical operations, showcasing the integration of AI systems with critical infrastructure.