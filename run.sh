#!/bin/bash

# This script runs the project

cd cloud_service_provider
# Run the cloud service provider
python3 main.py > /dev/null 2>&1 &
Cloud_PID=$!
if [ $? -eq 0 ]; then
    echo "Cloud server started successfully"
else
    echo "Cloud server failed to start"
    exit 1
fi

cd ../data_owner

# Run the data owner
python3 main.py > /dev/null 2>&1 &
Owner_PID=$!

if [ $? -eq 0 ]; then
    echo "Data owner started successfully"
else
    echo "Data owner failed to start"
    exit 1
fi

cd ..

sleep 5 # Wait for the data owner and Cloud to start

# upload the data from owner to Cloud
echo "Uploading data from owner to Cloud"
curl --location --request POST 'localhost:8080/upload'
echo ""
echo "Checking if decryption works"
curl --location --request GET 'localhost:8080/download'
echo ""
echo ""

# Run the client
cd query_user
python3 main.py

# Kill the data owner and Cloud
kill $Owner_PID
kill $Cloud_PID