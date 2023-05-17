from youtube_transcript_api import YouTubeTranscriptApi


def get_youtube_transcript(video_id):
    """
    Examples:
    subtitles disabled: https://www.youtube.com/watch?v=BMiNoO1DlD8
    """
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    cleaned_transcript = ' '. join([t['text'] for t in transcript]).lower().replace('>>', '')
    return cleaned_transcript