import os
from typing import Optional, Sequence

from google.cloud import videointelligence_v1 as vi
from datetime import timedelta

# see
# set up your cloud speech project. download credentials from IAM
GOOGLE_APPLICATION_CREDENTIALS="q:\git\\transcribeVideo\your-credentials.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

def transcribe_speech(
        video_uri: str,
        language_code: str,
        segments: Optional[Sequence[vi.VideoSegment]] = None,
) -> vi.VideoAnnotationResults:
    video_client = vi.VideoIntelligenceServiceClient()
    features = [vi.Feature.SPEECH_TRANSCRIPTION]
    config = vi.SpeechTranscriptionConfig(
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    context = vi.VideoContext(
        segments=segments,
        speech_transcription_config=config,
    )
    request = vi.AnnotateVideoRequest(
        input_uri=video_uri,
        features=features,
        video_context=context,
    )

    print(f'Processing video "{video_uri}"...')
    operation = video_client.annotate_video(request)

    return operation.result().annotation_results[0]  # Single video

def print_video_speech(results: vi.VideoAnnotationResults, min_confidence: float = 0.8):
    def keep_transcription(transcription: vi.SpeechTranscription) -> bool:
        return min_confidence <= transcription.alternatives[0].confidence

    transcriptions = results.speech_transcriptions
    transcriptions = [t for t in transcriptions if keep_transcription(t)]

    print(f" Speech Transcriptions: {len(transcriptions)} ".center(80, "-"))
    for transcription in transcriptions:
        first_alternative = transcription.alternatives[0]
        confidence = first_alternative.confidence
        transcript = first_alternative.transcript
        print(f" {confidence:4.0%} | {transcript.strip()}")

video_uri = "gs://cloud-samples-data/video/JaneGoodall.mp4"
language_code = "en-GB"

results = transcribe_speech(video_uri, language_code)
print_video_speech(results)
