#!/usr/bin/env python
# coding: utf-8

# In[2]:


# whisper /Users/cogsci-lasrlab1/Downloads/MFA_data/KidTalk/EB21_KT1/K1EB212participant_chick.WAV --model medium
# /Users/cogsci-lasrlab1/Desktop/Wisper_Work

# Importing Wisper
import whisper


# In[3]:


# Loading the Model
model = whisper.load_model("base")


# In[4]:


# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("chick.mp3")
audio = whisper.pad_or_trim(audio)


# In[5]:


# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)


# In[6]:


# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")


# In[7]:


# decode the audio
options = whisper.DecodingOptions(language="en", fp16 = False)
result = whisper.decode(model, mel, options)


# In[8]:


# print the recognized text
print(result.text)


# In[9]:


import os
from pydub import AudioSegment

# Converting wav files to MP3s
def wav_to_mp3(folder_path, output_file):
    if(os.path.isdir(folder_path)):
        try:
            files = os.listdir(folder_path)
            for file_name in files:
                if(".wav" in file_name):
                    # Input audio
                    input_wav_file = f'{folder_path}/{file_name}'
                    audio = AudioSegment.from_wav(input_wav_file)
                    
                    # Output the audio
                    output_mp3_file = f'{output_file}/{file_name[0:-4]}.mp3'
                    
                    # Export the audio
                    audio.export(output_mp3_file, format="mp3")
                    print(f"Conversion from {input_wav_file} to {output_mp3_file} completed.")
        except:
            print(f'Ups, theres an error')   
    else:
        print("The provided path is not a directory.")


# In[31]:


import os
import whisper  # Assuming this is your audio processing library

def list_files_in_folder(folder_path):
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)

        # Print header
        print(f"{'File':<30}{'Actual':<15}{'Predicted'}")
        print("="*70)  # Separator line

        for file_name in files:
            if('.mp3' in file_name or '.wav' in file_name):
                # Load model outside the loop to avoid reloading it for each file
                model = whisper.load_model("base")
                audio = whisper.load_audio(os.path.join(folder_path, file_name))
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                options = whisper.DecodingOptions(language="en", fp16=False)
                result = whisper.decode(model, mel, options)

                # Extract the actual word from the file name
                actual_word = file_name[19:-4]

                # Print formatted output
                print(f"{file_name:<30}{actual_word:<15}{result.text}")

    else:
        print("The provided path is not a directory.")

# Replace 'your_folder_path' with the actual path of the folder you want to open
folder_path = 'EB21_KT1_MP3/'
list_files_in_folder(folder_path)






# In[32]:


wav_to_mp3('EB21_KT1', 'EB21_KT1_MP3')
print()
list_files_in_folder('EB21_KT1_MP3/')


# In[33]:


wav_to_mp3('EB21_researcher_KT1', 'EB21_researcher_KT1_MP3')
print()
list_files_in_folder('EB21_researcher_KT1_MP3/')


# In[28]:


import os
import shutil

def separate_files(input_folder, output_folder_participant, output_folder_researcher):
    # Create output folders if they don't exist
    os.makedirs(output_folder_participant, exist_ok=True)
    os.makedirs(output_folder_researcher, exist_ok=True)

    # Iterate through files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if the file contains "participant" in its name
        if "participant" in filename.lower():
            destination_folder = output_folder_participant
        elif "researcher" in filename.lower():
            destination_folder = output_folder_researcher
        else:
            # Skip files that don't match the criteria
            continue

        # Move the file to the appropriate folder
        shutil.move(file_path, os.path.join(destination_folder, filename))

if __name__ == "__main__":
    # Replace these paths with your actual paths
    input_folder_path = "allKT1files/cb_ppts/"
    participant_output_path = "cb_ppts_participant/"
    researcher_output_path = "cb_ppts_researcher/"

    # Call the function to separate files
    separate_files(input_folder_path, participant_output_path, researcher_output_path)

    print("Separation completed.")


# In[34]:


#wav_to_mp3('EB21_researcher_KT1', 'EB21_researcher_KT1_MP3')
print()
list_files_in_folder('cb_ppts_participant/')


# In[ ]:




