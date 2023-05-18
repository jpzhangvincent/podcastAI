import os
import pandas as pd
import pickle
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def get_authenticated_service():
    api_key = os.getenv("GOOGLE_API_KEY")  # Replace with your API key
    return build("youtube", "v3", developerKey=api_key)


def get_youtube_transcript(video_id, postprocess=True):
    """
    Examples:
    subtitles disabled: https://www.youtube.com/watch?v=BMiNoO1DlD8
    """
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    if postprocess:
        cleaned_transcript = (
            " ".join([t["text"] for t in transcript]).lower().replace(">>", "")
        )
        return cleaned_transcript
    else:
        return transcript


def get_video_ids_from_collection_list(playlist_id):
    try:
        youtube = get_authenticated_service()
        next_page_token = None
        video_ids = []

        while True:
            # Call the API to retrieve videos from the collection list
            response = (
                youtube.playlistItems()
                .list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token,
                )
                .execute()
            )

            # Extract video IDs from the API response
            for item in response["items"]:
                video_id = item["contentDetails"]["videoId"]
                video_ids.append(video_id)

            next_page_token = response.get("nextPageToken")

            if not next_page_token:
                break

        return video_ids

    except HttpError as e:
        print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
    except Exception as e:
        print("An error occurred:\n%s" % str(e))


def get_video_metadata(video_id):
    try:
        youtube = get_authenticated_service()

        # Call the API to retrieve video details
        response = youtube.videos().list(part="snippet", id=video_id).execute()

        # Extract the metadata from the API response
        video_details = {
            "id": response["items"][0]["id"],
            "channelId": response["items"][0]["snippet"]["channelId"],
            "channelName": response["items"][0]["snippet"]["channelTitle"],
            "description": response["items"][0]["snippet"]["description"],
            "episodeTitle": response["items"][0]["snippet"]["title"],
            "tags": response["items"][0]["snippet"].get("tags"),
        }
        return video_details

    except HttpError as e:
        print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
    except Exception as e:
        print("An error occurred:\n%s" % str(e))


def get_all_episode_transcripts_by_playlist(playlist_id):
    video_ids = get_video_ids_from_collection_list(playlist_id)

    all_episodes_ls = []
    for video_id in video_ids:
        try:
            video_detail_dict = get_video_metadata(video_id)
            video_detail_dict["transcription"] = get_youtube_transcript(
                video_id, postprocess=False
            )
            all_episodes_ls.append(video_detail_dict)
        except:
            continue

    all_episodes_df = pd.DataFrame.from_records(all_episodes_ls)
    return all_episodes_df


def group_segments(segments, segment_max_length=600):
    grouped_segments = []
    for segment in segments:
        if len(grouped_segments) == 0:
            # add an end time
            # youtube transcript api doesn't return end times (start, duration)
            end_time = segment.get("start") + segment.get("duration")
            segment["end_time"] = end_time
            grouped_segments.append(segment)
        else:
            last_group = grouped_segments[-1]
            last_group_text = last_group.get("text")
            current_group_text = segment.get("text")
            combined_text = f"{last_group_text} {current_group_text}"
            if len(combined_text) <= segment_max_length:
                # add an end time
                # youtube transcript api doesn't return end times (start, duration)
                end_time = segment.get("start") + segment.get("duration")
                grouped_segments[-1]["end_time"] = end_time
                grouped_segments[-1]["text"] = combined_text
            else:
                # add an end time
                # youtube transcript api doesn't return end times (start, duration)
                end_time = segment.get("start") + segment.get("duration")
                segment["end_time"] = end_time
                grouped_segments.append(segment)
    return grouped_segments


def read_data_pickle(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


# get the list of videoids from the All-in podcast playlist
# collection_list_id = "PLn5MTSAqaf8peDZQ57QkJBzewJU1aUokl"  # Replace with your collection list ID
# video_ids = get_video_ids_from_collection_list(collection_list_id)
