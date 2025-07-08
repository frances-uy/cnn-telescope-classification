#!/bin/bash

# Check if logrotate is available and run it
if command -v logrotate &> /dev/null
then
    # Run logrotate with verbose output and force rotation
    logrotate -v -f /etc/logrotate.d/app_logs
else
    echo "logrotate not found, skipping log rotation"
fi

# Start the Python script
python3 real_time_classification_improved.py > /app/logs/classification.log 2>&1 &

# Start Gunicorn
gunicorn --bind 0.0.0.0:8080 web_app:app > /app/logs/gunicorn.log 2>&1
