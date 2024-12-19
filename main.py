import os
from pydub import AudioSegment, silence
import whisper
from pydub.utils import make_chunks


class AudioPreprocessor:
    def __init__(self, silenceThresh=-50, minSilenceLen=1000, keepSilence=500):
        self.silenceThresh = silenceThresh
        self.minSilenceLen = minSilenceLen
        self.keepSilence = keepSilence

    def splitOnSilence(self, inputFilePath):
        audio = AudioSegment.from_file(inputFilePath, format='m4a')
        chunks = silence.split_on_silence(audio,
            min_silence_len=self.minSilenceLen,
            silence_thresh=self.silenceThresh,
            keep_silence=self.keepSilence)
        return chunks
    
    def splitTwoMinuteChunks(self, inputFilePath):
        audio = AudioSegment.from_file(inputFilePath, format='m4a')
        chunkSize = 2 * 60 * 1000
        chunks = make_chunks(audio, chunkSize)
        return chunks

class Transcriber:
    def __init__(self, modelName='base', preprocessor=None,
                 transcriptFile='transcript.txt'):
        self.model = whisper.load_model(modelName)
        self.preprocessor = preprocessor
        self.transcriptFile = transcriptFile
       
        with open(self.transcriptFile, 'w') as f:
            f.write('')
       
    def transcribeWav(self, wavFilePath):
        result = self.model.transcribe(wavFilePath)
        return result['text']

    def processFile(self, inputFilePath):
        if self.preprocessor:
            print('Using preprocessor to split audio on silence.')
            chunks = self.preprocessor.splitOnSilence(inputFilePath)
        else:
            print('No preprocessor provided. Processing entire file.')
            audio = AudioSegment.from_file(inputFilePath, format='m4a')
            chunks = [audio]

        totalChunks = len(chunks)
        print(f'Total chunks: {totalChunks}')

        for i, chunk in enumerate(chunks, start=1):
            chunkWav = f'temp_chunk_{i}.wav'
            chunk.export(chunkWav, format='wav')
            text = self.transcribeWav(chunkWav)

            with open(self.transcriptFile, 'a') as tf:
                tf.write(text.strip() + '\n')

            os.remove(chunkWav)
            percentDone = (i / totalChunks) * 100
            print(f'Processed chunk {i}/{totalChunks} ({percentDone:.2f}%).')

        print('Transcription complete.')

if __name__ == '__main__':
    preprocessor = AudioPreprocessor(silenceThresh=-45, minSilenceLen=2000, keepSilence=500)
    transcriber = Transcriber(modelName='base', preprocessor=preprocessor, transcriptFile='transcript.txt')
    inputFilePath = '11-21-2024 Regular Session Board Meeting.m4a'
    transcriber.processFile(inputFilePath)
