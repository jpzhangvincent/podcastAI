# Podcast.ai
## Inspiration

The All In Podcast features David Sacks frequently saying "I didnt say that" when mentioning a misquote from the New York Times. 

## What it does

"I DIDN'T SAY THAT" parses youtube to perform topic modeling, and statement indexing that allows you to verify what you've said.

Currently, it's a chrome extension that overlays youtube.

[Demo](https://youtu.be/ECMRtjKdFnQ)

## How we built it

Technical implementation:
Youtube transcripts -> Embedding+ Index -> Topic modeling -> Prompt engineering/tuning -> Caching 

Deployment:
FastAPI to host langchain applications

UXUI:
Chrome extension to provide a seamless user experience
## Challenges we ran into

1. Collect high-quality transcript data with speaker tagging info for better parsing and summarization.
2. Not enough time, GPUs to train models in one day. We could fine-tune LLM models in parallel to better suit our use case. 

## Accomplishments that we're proud of

1. Collect and indexing ALL all-in podcast data and embeddings to support various NLP tasks, i.e question answering, semantic search and summarization
2. Prototype the chrome extension to improve the youtube/podcast watching and search experience
3. This could serve as a general fact checker tool for any media content.

## What we learned

The default behavior CHANGES EVERYTHING.

## What's next for I didnt say that! - the Youtube fact checker

We could run this live across all podcasts and during contentious topics and Presidential debates

# Backend API setup
```
# install python libraries
pip install -r src/requirements.txt

# running the fastAPI server in local
lc-serve deploy local podcast_api
```

# Frontend UI(Extension)
Please refer to this [repo](https://github.com/theptrk/ididntsaythat)


