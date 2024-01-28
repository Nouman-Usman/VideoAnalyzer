import os
from googleapiclient.discovery import build

# Set your API key
api_key = "API Key"

# Create a YouTube API client
youtube = build('youtube', 'v3', developerKey=api_key)

# Function to search on YouTube
def search_youtube(query, max_results=5):
    request = youtube.search().list(
        q=query,
        part='id,snippet',
        type='video',
        maxResults=max_results
    )
    response = request.execute()

    # Extract video details from the response
    videos = []
    for item in response.get('items', []):
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        videos.append({'title': video_title, 'video_id': video_id, 'video_url': video_url})

    return videos


# search_query = "Python programming tutorial"
# results = search_youtube(search_query)


def printResult(query):
    results = search_youtube(query)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} - Video ID: {result['video_id']}")
        print(f"   Video URL: {result['video_url']}")
