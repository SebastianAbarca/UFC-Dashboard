import os
import time
import re
import pandas as pd
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

# --- Configuration ---
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
UFC_CHANNEL_ID = "UCvgfXK4nTYKudb0rFR6noLA"  # UFC's official channel

# --- Regex to match UFC Embedded ---
pattern = r"UFC (\d+) Embedded: Vlog Series - Episode (\d+)"

# --- Authenticate and get service ---
def get_authenticated_service():
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_local_server(port=0)
    return googleapiclient.discovery.build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

# --- Get Uploads Playlist ID ---
def get_uploads_playlist_id(youtube, channel_id):
    response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()
    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

# --- Save DataFrame to CSV ---
def save_to_csv(video_data, filename="ufc_video_data.csv"):
    df = pd.DataFrame(video_data)
    df.to_csv(filename, index=False)
    print(f"\nSaved {len(df)} videos to {filename}")

# --- Main ---
if __name__ == "__main__":
    youtube = get_authenticated_service()
    uploads_playlist_id = get_uploads_playlist_id(youtube, UFC_CHANNEL_ID)

    video_data = []
    next_page_token = None
    total_fetched = 0

    print("Fetching video titles and view counts from UFC channel...")

    while True:
        try:
            # Get video IDs from the uploads playlist
            playlist_response = youtube.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()

            video_ids = [item["snippet"]["resourceId"]["videoId"] for item in playlist_response["items"]]

            # Get video details (title, viewCount)
            video_response = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(video_ids)
            ).execute()

            for item in video_response["items"]:
                title = item["snippet"]["title"]
                views = int(item["statistics"].get("viewCount", 0))
                match = re.search(pattern, title)
                if match:
                    event = int(match.group(1))
                    episode = int(match.group(2))
                else:
                    event = episode = None
                video_data.append({
                    "Title": title,
                    "Views": views,
                    "Event": event,
                    "Episode": episode
                })
                total_fetched += 1

            print(f"Fetched {total_fetched} videos...")

            next_page_token = playlist_response.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(0.5)

        except googleapiclient.errors.HttpError as e:
            if e.resp.status == 429:
                print("Rate limit reached (HTTP 429). Saving data collected so far...")
                save_to_csv(video_data)
                break
            else:
                print(f"API error: {e}")
                save_to_csv(video_data)
                break
        except Exception as e:
            print(f"Unexpected error: {e}")
            save_to_csv(video_data)
            break

    # Final save (in case loop ends naturally)
    if video_data:
        save_to_csv(video_data)
    print(f"\nTotal videos fetched: {len(video_data)}")

