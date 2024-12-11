import os
import torch
import numpy as np
import scipy.io.wavfile
import google.generativeai as genai
from transformers import Wav2Vec2ForCTC, AutoProcessor, VitsModel, AutoTokenizer
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tempfile

# Configuration Gemini
GOOGLE_API_KEY = ''  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel('gemini-1.5-pro')

# ASR Model Configuration
model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
processor.tokenizer.set_target_lang("fon")
model.load_adapter("fon")

# TTS Model for Fon
fon_tts_model = VitsModel.from_pretrained("facebook/mms-tts-fon")
fon_tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fon")

class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def add_interaction(self, user_input, bot_response):
        self.history.append({
            'user': user_input,
            'bot': bot_response
        })
        # Truncate history if necessary
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context(self):
        # Format history for conversation context
        context = "\nConversation History:\n"
        for interaction in self.history:
            context += f"User: {interaction['user']}\n"
            context += f"Assistant: {interaction['bot']}\n"
        return context

# Conversation memory instance
conversation_memory = ConversationMemory()

def transcribe_audio(audio_path):
    """Transcribe audio with Fon ASR model"""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Preprocess audio
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16_000, return_tensors="pt")

        # Transcribe
        with torch.no_grad():
            outputs = model(**inputs).logits

        # Decode logits
        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)
        
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

def generate_response(user_input):
    """Generate response with Gemini"""
    try:
        # Add conversation history context
        context = conversation_memory.get_context()

        # Response generation prompt
        prompt = f"""
        You are a conversational assistant in the Fongbe language.
        Respond naturally and concisely in everyday language.

        {context}

        User's last input: {user_input}
        Respond in Fongbe.
        """

        response = chat_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation error: {str(e)}")

def text_to_speech(text):
    """Convert text to speech in Fon"""
    try:
        inputs = fon_tts_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = fon_tts_model(**inputs).waveform

        output = output.cpu()
        data_np = output.numpy()
        data_np_squeezed = np.squeeze(data_np)

        # Audio normalization
        data_np_normalized = np.int16(data_np_squeezed * 32767)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            scipy.io.wavfile.write(
                temp_output.name, 
                rate=fon_tts_model.config.sampling_rate, 
                data=data_np_normalized
            )
            return temp_output.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text-to-speech error: {str(e)}")

# Create FastAPI app
app = FastAPI(title="Fongbe Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/chat/audio")
async def process_audio_interaction(audio: UploadFile = File(...)):
    """Process audio interaction endpoint"""
    try:
        # Save uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
            temp_input.write(await audio.read())
            temp_input_path = temp_input.name

        # Transcribe audio
        transcription = transcribe_audio(temp_input_path)

        # Generate response
        response = generate_response(transcription)

        # Update conversation memory
        conversation_memory.add_interaction(transcription, response)

        # Convert response to speech
        audio_response_path = text_to_speech(response)

        # Return audio file
        return FileResponse(
            audio_response_path, 
            media_type="audio/wav", 
            filename="response.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        for temp_file in [temp_input_path, audio_response_path]:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)