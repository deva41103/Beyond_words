from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import spacy
import subprocess
import speech_recognition as sr
from werkzeug.utils import secure_filename
import atexit
import torch
import numpy as np
from model import Gloss2PoseTransformer as Gloss2PoseModel  # import your model class
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "supersecretkey"

# Create upload folders if they don't exist
UPLOAD_FOLDER = os.path.join("static", "videos")
AUDIO_FOLDER = os.path.join("static", "audio")
KEYPOINT_FOLDER = os.path.join("static", "keypoints")
os.makedirs(KEYPOINT_FOLDER, exist_ok=True)
app.config["KEYPOINT_FOLDER"] = KEYPOINT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["AUDIO_FOLDER"] = AUDIO_FOLDER

# Try to find ffmpeg in system PATH
FFMPEG_PATH = "ffmpeg"  # This will use the system ffmpeg if available
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

db = SQLAlchemy(app)

# Load spaCy for POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# User Model
class User(db.Model):
    username = db.Column(db.String(20), primary_key=True)
    emailid = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    contact = db.Column(db.String(15), nullable=False)
    education = db.Column(db.Text, nullable=False)
    gender = db.Column(db.String(5), nullable=False)
    age = db.Column(db.String(5), nullable=False)

# Video Model
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    video_url = db.Column(db.String(255), nullable=False)
    audio_url = db.Column(db.String(255), nullable=True)
    transcript = db.Column(db.Text, nullable=True)
    gloss_text = db.Column(db.Text, nullable=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        try:
            user = User(
                username=request.form["username"],
                password=request.form["password"],
                emailid=request.form["email"],
                name=request.form["name"],
                contact=request.form["contact"], 
                education=request.form["education"], 
                gender=request.form["gender"],
                age=request.form["age"]
            )
            db.session.add(user)
            db.session.commit()
            return redirect(url_for("login"))
        except Exception as e:
            db.session.rollback()
            return render_template('signup.html', error=str(e))
    return render_template('signup.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and user.password == request.form["password"]:
            session["username"] = user.username
            return redirect(url_for("dashboard"))
        return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path, output_audio_path):
    try:
        subprocess.run(
            [FFMPEG_PATH, "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_audio_path],
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        return f"Error in speech recognition: {str(e)}"

def rule_based_transformation(text):
    try:
        doc = nlp(text)
        gloss_tokens = []
        for token in doc:
            if token.pos_ in {"DET", "ADP", "AUX", "PRON"}:  # Removing unnecessary words
                continue
            if token.dep_ == "neg":
                gloss_tokens.append("NOT")
                continue
            if token.pos_ == "VERB" and token.head.pos_ == "ROOT":
                gloss_tokens.insert(0, token.lemma_)
            else:
                gloss_tokens.append(token.lemma_)
        return " ".join(gloss_tokens).upper()
    except Exception as e:
        print(f"Error in gloss transformation: {str(e)}")
        return text.upper()  # Fallback to simple uppercase if transformation fails


# Load vocab
with open('vocab1.json', 'r') as f:
    gloss2idx = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pose_dim = 1629  # FULL pose output size (33+468+21+21) * 3
model = Gloss2PoseModel(
    vocab_size=len(gloss2idx),
    pose_dim=pose_dim,
    hidden=256,
    num_layers=4,
    nhead=4
)

model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

@app.route("/generate_3d_sign", methods=["GET", "POST"])
def generate_3d_sign():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            title = request.form.get("title", "").strip()
            description = request.form.get("description", "").strip()
            file = request.files.get("video")

            if not title:
                return render_template("generate_3d_sign.html", error="Title is required")
            if not file or file.filename == "":
                return render_template("generate_3d_sign.html", error="No file selected")
            if not allowed_file(file.filename):
                return render_template("generate_3d_sign.html", error="Invalid file type")

            # Save video file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file_url = f"/static/videos/{filename}"
            file.save(filepath)

            # Extract audio
            audio_filename = filename.rsplit(".", 1)[0] + ".wav"
            audio_path = os.path.join(app.config["AUDIO_FOLDER"], audio_filename)
            audio_url = f"/static/audio/{audio_filename}"

            if not extract_audio(filepath, audio_path):
                os.remove(filepath)
                return render_template("generate_3d_sign.html", error="Failed to extract audio")

            # Speech to text and rule-based gloss
            transcript = speech_to_text(audio_path)
            rule_based_gloss = rule_based_transformation(transcript)
            gloss_tokens = rule_based_gloss.split()
            gloss_indices = [gloss2idx.get(token, gloss2idx.get("<unk>", 0)) for token in gloss_tokens]

            # Autoregressive inference (like generate_pose)
            model.eval()
            pose_sequence = []

            for gloss_id in gloss_indices:
                gloss_tensor = torch.tensor([gloss_id], dtype=torch.long).to(device)  # [1]
                tgt_seq = torch.zeros((1, 1, model.pose_dim), dtype=torch.float32).to(device)  # [1, 1, D]

                with torch.no_grad():
                    for _ in range(28):  # Generate 6 frames per gloss token
                        pred = model(gloss_tensor, tgt_seq)  # [1, 1, pose_dim]
                        pose_sequence.append(pred[0, 0, :].cpu().numpy())
                        tgt_seq = pred  # Autoregressive step

            # Save keypoints
            predicted_keypoints = np.array(pose_sequence)  # [T, D]
            keypoints_filename = filename.rsplit(".", 1)[0] + "_keypoints.npy"
            keypoints_path = os.path.join("static", "keypoints", keypoints_filename)
            os.makedirs("static/keypoints", exist_ok=True)
            np.save(keypoints_path, predicted_keypoints)

            print("📄 Transcript:", transcript)
            print("🧠 Gloss:", rule_based_gloss)
            print("📊 Output shape:", predicted_keypoints.shape)
            print("📍 Keypoints saved to:", keypoints_path)

            # Save to database
            new_video = Video(
                title=title,
                description=description,
                video_url=file_url,
                audio_url=audio_url,
                transcript=transcript,
                gloss_text=rule_based_gloss
            )
            db.session.add(new_video)
            db.session.commit()
            print("✅ Video entry saved to DB")

            return redirect(url_for("pretrained"))

        except Exception as e:
            db.session.rollback()
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
            return render_template("generate_3d_sign.html", error=f"An error occurred: {str(e)}")

    return render_template("generate_3d_sign.html")



@app.route("/pretrained")
def pretrained():
    if "username" not in session:
        return redirect(url_for("login"))

    videos = Video.query.all()
    updated_videos = []

    for video in videos:
        # Get the video filename from the video_url
        filename = os.path.basename(video.video_url)
        video_url = video.video_url
        
        # Generate corresponding keypoint path
        keypoint_filename = filename.rsplit(".", 1)[0] + "_keypoints.npy"
        keypoint_path = os.path.join(app.config["KEYPOINT_FOLDER"], keypoint_filename)
        has_gloss = os.path.exists(keypoint_path)

        updated_videos.append({
            "id": video.id,
            "title": video.title,
            "description": video.description,
            "video_url": video_url,
            "has_gloss": has_gloss,
            "keypoint_url": f"/static/keypoints/{keypoint_filename}" if has_gloss else None
        })

    return render_template("pretrained.html", 
                         username=session["username"], 
                         videos=updated_videos)


@app.route('/keypoints/<filename>')
def serve_keypoints(filename):
    keypoints_path = os.path.join("static", "keypoints", filename)
    if not os.path.exists(keypoints_path):
        return jsonify({"error": "Keypoints file not found"}), 404

    data = np.load(keypoints_path).tolist()
    return jsonify(data)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/get_skeleton_data/<video_id>')
def get_skeleton_data(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({"error": "Video not found"}), 404

    filename = os.path.basename(video.video_url)
    keypoint_filename = filename.rsplit(".", 1)[0] + "_keypoints.npy"
    keypoints_path = os.path.join(app.config["KEYPOINT_FOLDER"], keypoint_filename)

    if not os.path.exists(keypoints_path):
        return jsonify({"error": "Keypoints not found"}), 404

    keypoints = np.load(keypoints_path)
    frames = []

    for frame in keypoints:
        pose = frame[:33*3].reshape(-1, 3).tolist()
        face = frame[33*3:(33+468)*3].reshape(-1, 3).tolist()
        lh = frame[(33+468)*3:(33+468+21)*3].reshape(-1, 3).tolist()
        rh = frame[(33+468+21)*3:].reshape(-1, 3).tolist()

        frames.append({
            "pose": pose,
            "face": face,
            "left_hand": lh,
            "right_hand": rh
        })

    return jsonify({"frames": frames, "fps": 30})


@app.route('/skeleton_preview/<video_id>')
def skeleton_preview(video_id):
    video = Video.query.get(video_id)
    if not video:
        return "Video not found", 404
    
    filename = os.path.basename(video.video_url)
    keypoint_filename = filename.rsplit(".", 1)[0] + "_keypoints.npy"
    keypoints_path = os.path.join(app.config["KEYPOINT_FOLDER"], keypoint_filename)
    
    if not os.path.exists(keypoints_path):
        return "Keypoints not found", 404
    
    keypoints = np.load(keypoints_path)
    return render_template('skeleton_preview.html', 
                         video_url=video.video_url,
                         video_id=video_id,
                         username=session.get("username"))



if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)