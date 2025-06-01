#!/bin/bash

# Navigate to the project directory
cd /moodify_api

# Pull latest changes from git
echo "Pulling latest changes from git..."
git pull origin main

# Install any new dependencies
echo "Updating dependencies..."
pip install -r requirements.txt

# Reload systemd daemon and restart services
echo "Restarting services..."
sudo systemctl daemon-reload
sudo systemctl restart gunicorn
sudo systemctl restart nginx

echo "Update completed successfully!"