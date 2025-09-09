from app import db, Video

with db.session.begin():
    db.session.add(Video(title="ASL Basics", description="Basic ASL signs", video_url="/static/videos/asl_basics.mp4"))
    db.session.add(Video(title="Alphabet Signs", description="Learn the ASL alphabet", video_url="/static/videos/asl_alphabet.mp4"))
    db.session.commit()
