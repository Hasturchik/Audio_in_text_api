import json
import os
import subprocess
import wave
import numpy as np
from vosk import Model, KaldiRecognizer
from pyAudioAnalysis import audioSegmentation as aS
import librosa
import tempfile


MODEL_PATH = "vosk-model-small-ru-0.22"
model = Model(MODEL_PATH)


def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Конвертирует MP3 файл в WAV формат.
    """
    command = ["ffmpeg", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path]
    subprocess.run(command, check=True)


def recognize_speech(wav_path):
    """
    Распознаёт речь в файле.
    """
    results = []
    with wave.open(wav_path, "rb") as wf:
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        while True:
            data = wf.readframes(2000)
            if len(data) == 0:
                final_result = rec.FinalResult()
                if final_result:
                    results.append(json.loads(final_result))
                break
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result()))
    return results


def diarize_audio(wav_path, n_speakers=2):
    """
    Определяет, кто говорит в аудио файле (диаризация).
    """
    flags, _, _ = aS.speaker_diarization(wav_path, n_speakers=n_speakers)
    return flags


def analyze_pitch_librosa(wav_path, start_time, end_time):
    """
    Анализирует интонацию в отрезке аудио.
    :param wav_path: Путь к WAV файлу.
    :param start_time: Начальное время сегмента.
    :param end_time: Конечное время сегмента.
    :return: True, если интонация высока (порог 200 Гц), иначе False.
    """
    signal, sr = librosa.load(wav_path, sr=None, offset=start_time, duration=end_time - start_time)
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
    pitch_median = np.median(pitches[pitches > 0])
    return bool(pitch_median > 200)


def analyze_gender_librosa(wav_path, start_time, end_time):
    """
    Анализирует пол говорящего на основе интонации.
    :param wav_path: Путь к WAV файлу.
    :param start_time: Начальное время сегмента.
    :param end_time: Конечное время сегмента.
    :return: Пол ("male", "female") или "unknown", если не удалось определить.
    """
    signal, sr = librosa.load(wav_path, sr=None, offset=start_time, duration=end_time - start_time)
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
    pitch_median = np.median(pitches[pitches > 0])
    if np.isnan(pitch_median):
        return "unknown"
    return "male" if pitch_median < 165 else "female"


def merge_results(speech_results, diarization_flags, wav_path):
    """
    Объединяет результаты распознавания речи и диаризации для получения финальной расшифровки.
    :param speech_results: Результаты распознавания речи.
    :param diarization_flags: Результаты диаризации (спикеры).
    :param wav_path: Путь к WAV файлу.
    :return: Словарь с итоговыми результатами диалога и длительностью для каждого спикера.
    """
    dialog = []
    result_duration = {"receiver": 0, "transmitter": 0}
    speaker_mapping = {0: "receiver", 1: "transmitter"}

    current_speaker = diarization_flags[0]
    current_text = ""
    current_duration = 0
    previous_time = 0
    speaker_gender = {}

    for result in speech_results:
        if "result" in result:
            words = result["result"]
            for word in words:
                start_time = word["start"]
                end_time = word["end"]
                word_speaker = diarization_flags[int((start_time + end_time) // 2)]

                if word_speaker != current_speaker:
                    if current_text.strip():
                        pitch_analysis = analyze_pitch_librosa(wav_path, previous_time, start_time)
                        if current_speaker not in speaker_gender:
                            speaker_gender[current_speaker] = analyze_gender_librosa(wav_path, previous_time,
                                                                                     start_time)

                        dialog.append({
                            "source": speaker_mapping[current_speaker],
                            "text": current_text.strip(),
                            "duration": current_duration,
                            "intonation_high": pitch_analysis,
                            "gender": speaker_gender[current_speaker]
                        })
                        result_duration[speaker_mapping[current_speaker]] += current_duration

                    current_speaker = word_speaker
                    current_text = ""
                    current_duration = 0

                current_text += f" {word['word']}"
                current_duration += end_time - start_time
                previous_time = start_time

    if current_text.strip():
        pitch_analysis = analyze_pitch_librosa(wav_path, previous_time, end_time)
        if current_speaker not in speaker_gender:
            speaker_gender[current_speaker] = analyze_gender_librosa(wav_path, previous_time, end_time)

        dialog.append({
            "source": speaker_mapping[current_speaker],
            "text": current_text.strip(),
            "duration": current_duration,
            "intonation_high": pitch_analysis,
            "gender": speaker_gender[current_speaker]
        })
        result_duration[speaker_mapping[current_speaker]] += current_duration

    return {"dialog": dialog, "result_duration": result_duration}


def process_audio(file):
    """
    Обрабатывает аудио файл (MP3), выполняя конвертацию в WAV, распознавание речи и диаризацию.
    :param file: Аудиофайл в формате MP3.
    :return: Результаты обработки в формате JSON.
    """
    try:
        temp_path = tempfile.mktemp(suffix='.mp3')
        wav_path = tempfile.mktemp(suffix='.wav')

        with open(temp_path, 'wb') as f:
            f.write(file.read())

        convert_mp3_to_wav(temp_path, wav_path)
        diarization_flags = diarize_audio(wav_path)
        speech_results = recognize_speech(wav_path)
        output = merge_results(speech_results, diarization_flags, wav_path)

        return output

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)