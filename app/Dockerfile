FROM rockylinux:9

# Set the working directory in the container
WORKDIR /app

# Install EPEL repository and system dependencies
RUN dnf -y install epel-release && \
    dnf -y update && \
    dnf -y install \
    python3 \
    python3-pip \
    procps-ng \
    net-tools \
    iputils \
    wget \
    vim \
    nano \
    git \
    sudo \
    less \
    man-db \
    gcc \
    gcc-c++ \
    make \
    openssl-devel \
    zlib-devel \
    readline-devel \
    sqlite-devel \
    bind-utils \
    htop \
    logrotate \
    && dnf clean all

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Create directories for saving results, processed images, and logs
RUN mkdir -p /app/cloud_classification_data/processed_images /app/logs

# Give appropriate permissions
RUN chmod -R 755 /app

# Set up log rotation
RUN echo $'/app/logs/*.log {\n\
    size 20M\n\
    rotate 5\n\
    missingok\n\
    notifempty\n\
    compress\n\
    delaycompress\n\
    copytruncate\n\
}' > /etc/logrotate.d/app_logs

# Ensure the script has execute permissions
RUN chmod +x /app/start.sh

# Run start.sh when the container launches
CMD ["/bin/bash", "/app/start.sh"]
