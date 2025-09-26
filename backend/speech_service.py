import speech_recognition as sr
import io
import base64
from pydub import AudioSegment
import tempfile
import os

class SpeechService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
    def convert_audio_format(self, audio_data, input_format='wav'):
        """Convert audio data to WAV format if needed"""
        try:
            if input_format.lower() != 'wav':
                # Convert using pydub
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)
                wav_io = io.BytesIO()
                audio.export(wav_io, format='wav')
                return wav_io.getvalue()
            return audio_data
        except Exception as e:
            print(f"Audio conversion error: {e}")
            return audio_data
    
    def speech_to_text(self, audio_data, audio_format='wav', language='en-US'):
        """Convert speech audio to text using multiple fallback methods"""
        try:
            # Convert audio to WAV if needed
            if audio_format.lower() != 'wav':
                audio_data = self.convert_audio_format(audio_data, audio_format)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio.flush()
                
                # Use speech_recognition library
                with sr.AudioFile(temp_audio.name) as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.record(source)
                
                # Try Google Speech Recognition first
                try:
                    text = self.recognizer.recognize_google(audio, language=language)
                    print("✅ Google Speech Recognition successful")
                    return text.strip()
                except sr.UnknownValueError:
                    print("⚠️ Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print(f"⚠️ Google Speech Recognition error: {e}")
                
                # Fallback: Try Sphinx (offline)
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    print("✅ CMU Sphinx recognition successful")
                    return text.strip()
                except sr.UnknownValueError:
                    print("⚠️ Sphinx could not understand audio")
                except Exception as e:
                    print(f"⚠️ Sphinx recognition error: {e}")
            
            return None
            
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return None
        finally:
            # Clean up temporary file
            try:
                if 'temp_audio' in locals():
                    os.unlink(temp_audio.name)
            except:
                pass
    
    def is_audio_content(self, base64_data):
        """Check if the base64 data is audio content"""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(base64_data)
            
            # Try to read as audio file
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                return True, len(audio_bytes), audio.channels, audio.frame_rate
            except:
                return False, 0, 0, 0
                
        except Exception as e:
            print(f"Error checking audio content: {e}")
            return False, 0, 0, 0

# Global instance
speech_service = SpeechService()