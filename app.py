#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, jsonify
import pickle
from twilio.rest import Client
import os
from skimage import io, color, transform
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# In[2]:


app = Flask(__name__)


# In[3]:


# %run StatisticalModeling.ipynb


# In[4]:


# Twilio credentials
account_sid = 'ACed01a9eeb39c2db1219e854f01a82054'
auth_token = 'ac5c6cd2d347bae0c5d73252a5e96e25'
twilio_phone_number = '+12053465799'
user_phone_number = '+918950910576'

client = Client(account_sid, auth_token)


# In[5]:


def Z_Score_Value(Value):
    mean = 158668.18606839626
    std = 264941.5785563747
    Z_score = (Value - mean)/std
    Threshold_value = 12.40
    
    if Z_score >= Threshold_value:
        return 1
    else:
        return 0
    
def Tukey_Fences_Values(Value):
    Q1 = 12149.490000000002
    Q3 = 213762.15000000002
    IQR = Q3 - Q1
    tukey_threshold = 16.00 * IQR
    
    if Value < Q1 - tukey_threshold or  Value > Q3 + tukey_threshold:
        return 1
    else:
        return 0
    
    
def modified_zscore_values(value):
    median = 76345.78
    median_absolute_deviation = np.median(np.abs(value - median))
    modified_z_score = np.abs(0.6745 * (value - median) / median_absolute_deviation)
    threshold = 32.30
    
    if modified_z_score > threshold:
        return 1
    else:
        return 0


# In[6]:


@app.route('/')
def index():
    return render_template('index.html')


# In[7]:


@app.route('/predict_anomaly', methods=['POST'])
def predict_anomaly():
    try:
        selected_function = request.json['selectedFunction']
        payment_type = request.json['paymentType']
        old_balance = request.json['oldBalance']
        amount = request.json['amount']

        # Choose the selected function
        if selected_function == 'modified_zscore':
            prediction = modified_zscore_values(amount)
        elif selected_function == 'tukey_fences':
            prediction = Tukey_Fences_Values(old_balance)
        elif selected_function == 'z_score':
            prediction = Z_Score_Value(amount)
        else:
            raise ValueError('Invalid function selected')

        # Simulate updating the balance (replace with your actual logic)
        current_balance = old_balance - amount if payment_type == 'debit' else old_balance + amount
        
         # Send SMS if prediction is 1 (anomaly)
        if prediction == 1:
            send_sms(f'Fraud detected! Payment Type: {payment_type}, Old Balance: {old_balance}, Amount: {amount}')

        return jsonify({'prediction': prediction, 'currentBalance': current_balance})

    except Exception as e:
        return jsonify({'error': str(e)})
    

def send_sms(message):
    try:
        message = client.messages.create(
            to=user_phone_number,
            from_=twilio_phone_number,
            body=message
        )
        print(f"Message sent successfully. SID: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")


# In[8]:


with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    


# In[9]:


@app.route('/machinelearning')
def machinelearning():
    return render_template('machinelearning.html')

@app.route('/machinelearningpredict', methods=['POST'])
def machinelearningpredict():
    type_of_payment = request.form['type_of_payment']
    old_balance = float(request.form['old_balance'])
    amount = float(request.form['amount'])

    # Map type_of_payment to encoded values
    payment_mapping = {'cash_out': 1, 'payment': 3, 'cash_in': 0, 'transfer': 4, 'debit': 2}
    encoded_payment = payment_mapping.get(type_of_payment.lower(), 0)

    # Create new_balance variable
    new_balance = old_balance + amount

    # Make prediction
    prediction = model.predict([[encoded_payment, amount, old_balance, new_balance]])

    # Display the result on the HTML page
    if prediction[0] == 0:
        result = 'Not Fraud'
    else:
        result = 'Fraud Transaction'
        send_sms(f'Fraud detected! Payment Type: {type_of_payment}, Old Balance: {old_balance}, Amount: {amount}')
    return render_template('machinelearning.html', result=result)

def send_sms(message):
    try:
        message = client.messages.create(
            to=user_phone_number,
            from_=twilio_phone_number,
            body=message
        )
        print(f"Message sent successfully. SID: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")


# In[10]:


from PIL import Image

def is_anomaly(image_path):
    # Open the image using Pillow
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_data = np.array(image)

    # Flatten the 2D array to 1D for simplicity (modify based on your actual data structure)
    flattened_data = image_data.flatten()

    # Calculate Z-score for each pixel
    z_scores = (flattened_data - np.mean(flattened_data)) / np.std(flattened_data)
    
    threshold = 0.3547328492

    # Set anomaly flag based on threshold
    anomalies = np.abs(z_scores) > threshold
    return anomalies

@app.route('/detect_anomaly', methods=['GET', 'POST'])

def detect_anomaly():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Read image data
        if image_file:
            # Save the uploaded image to a temporary file
            image_path = "temp_image.jpg"  # Change the extension based on the uploaded image format
            image_file.save(image_path)

            # Check for anomalies
            anomalies = is_anomaly(image_path)

            # Render the result on the web page
            return render_template('imageres.html', anomalies=anomalies.any())

        # If no image file is provided
        return render_template('imageind.html', error="Please upload an image.")

    # If the request method is 'GET'
    return render_template('imageind.html')


# In[ ]:


if __name__ == '__main__':
    app.run(host="0.0.0.0")


# In[ ]:




